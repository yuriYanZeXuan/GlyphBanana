"""Qwen3-VL reward server using OpenAI API."""

import argparse
import base64
import io
import os
import re
from contextlib import asynccontextmanager

import Levenshtein
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI
from openai import OpenAI
from PIL import Image
from pydantic import BaseModel

from debug_utils import save_debug

# Load API keys from .env
load_dotenv()

# System prompts
VLM_PROMPT = """Rate the aesthetic quality of this image (1-5):
1: Very poor (blurry, bad exposure, messy)
2: Poor (noticeable issues)
3: Fair (decent but average)
4: Good (sharp, good composition)
5: Excellent (masterful, impactful)

Provide analysis in <Thought> tags, then score in <Score> tags.
<Thought>[analysis]</Thought>
<Score>X</Score>"""

OCR_PROMPT = """Read ALL text visible in this image.
Output only the raw text content, nothing else.
If multiple text elements, separate with spaces.
Do not add explanations."""


class ScoreRequest(BaseModel):
    image: str
    prompt: str
    mask: str | None = None
    timestep: str | None = None
    text_gt: str = ""  # Ground truth text for OCR accuracy


# Global client
client: OpenAI | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Qwen3 reward server starting...")
    yield
    print("Qwen3 reward server shutting down...")


app = FastAPI(lifespan=lifespan)


def encode(img: Image.Image) -> str:
    """Encode image to base64 for API."""
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def extract_vlm_score(text: str) -> float:
    """Extract score from 1-5 and normalize to 0-1."""
    m = re.search(r"<Score>(\d)</Score>", text)
    return float(m.group(1)) / 5.0 if m else 0.0


def compute_ocr_acc(pred: str, gt: str) -> float:
    """Compute OCR accuracy using Levenshtein distance."""
    pred_clean = pred.replace(" ", "").lower()
    gt_clean = gt.replace(" ", "").lower()

    if not gt_clean:
        return 1.0 if not pred_clean else 0.0
    if not pred_clean:
        return 0.0

    dist = Levenshtein.distance(pred_clean, gt_clean)
    return max(0.0, 1.0 - dist / len(gt_clean))


def score_vlm(img: Image.Image) -> float:
    """Get aesthetic score from Qwen3-VL."""
    img_b64 = encode(img)

    resp = client.chat.completions.create(
        model="qwen3-vl-235b-a22b-instruct",
        messages=[
            {"role": "system", "content": VLM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
                    {"type": "text", "text": "Rate this image quality."},
                ],
            },
        ],
        max_tokens=512,
        temperature=0.0,
    )

    return extract_vlm_score(resp.choices[0].message.content)


def score_ocr(img: Image.Image, gt: str) -> tuple[str, float]:
    """Get OCR text and accuracy from Qwen3-VL."""
    img_b64 = encode(img)

    resp = client.chat.completions.create(
        model="qwen3-vl-235b-a22b-instruct",
        messages=[
            {"role": "system", "content": OCR_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
                    {"type": "text", "text": "Read all text in this image."},
                ],
            },
        ],
        max_tokens=256,
        temperature=0.0,
    )

    text = resp.choices[0].message.content.strip()
    acc = compute_ocr_acc(text, gt) if gt else 0.0

    return text, acc


@app.post("/score")
async def score(req: ScoreRequest):
    """Score image with VLM aesthetic and OCR accuracy."""
    # Decode image
    img_bytes = base64.b64decode(req.image)
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    # Decode and apply mask
    masked = img
    if req.mask:
        mask_bytes = base64.b64decode(req.mask)
        mask = Image.open(io.BytesIO(mask_bytes)).convert("L")
        if mask.size != img.size:
            mask = mask.resize(img.size, Image.NEAREST)

        import numpy as np
        marr = (np.array(mask) > 127).astype(np.uint8)
        iarr = np.array(img)
        iarr[marr == 0] = 255
        masked = Image.fromarray(iarr)

    # Extract ground truth from prompt if not provided
    gt = req.text_gt
    if not gt:
        m = re.search(r'["""]([^"""]+)["""]', req.prompt)
        if m:
            gt = m.group(1)

    # Score
    vlm = score_vlm(img)
    ocr_text, ocr = score_ocr(masked, gt)

    save_debug(img, masked, req.prompt, vlm, ocr, ocr_text, timestep=req.timestep)

    return {
        "vlm_score": vlm,
        "ocr_score": ocr,
        "ocr_text": ocr_text,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    global client
    client = OpenAI(
        api_key=os.getenv("QST_API_KEY"),
        base_url=os.getenv("QST_BASE_URL"),
    )

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
