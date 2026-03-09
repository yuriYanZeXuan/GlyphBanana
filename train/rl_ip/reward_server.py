"""Reward server providing VLM and OCR scores via HTTP API."""

import argparse
import base64
import io
import os
from contextlib import asynccontextmanager

import numpy as np
import uvicorn
from fastapi import FastAPI
from PIL import Image
from pydantic import BaseModel

from debug_utils import save_debug
from ocr import OCRScorer
from qwenvl import QwenScorer


class ScoreRequest(BaseModel):
    image: str
    prompt: str
    mask: str | None = None
    timestep: str | None = None


# Global scorer instances
qwen: QwenScorer | None = None
ocr: OCRScorer | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Reward server starting...")
    yield
    print("Reward server shutting down...")


app = FastAPI(lifespan=lifespan)


@app.post("/score")
async def score(req: ScoreRequest):
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "N/A")
    print(f"Scoring on GPU {gpu}: {req.prompt[:30]}...")

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

        mask_arr = (np.array(mask) > 127).astype(np.uint8)
        img_arr = np.array(img)
        img_arr[mask_arr == 0] = 255
        masked = Image.fromarray(img_arr)

    # Score
    vlm = qwen.score(img, req.prompt)
    text, conf = ocr.score(masked)

    print(f"  VLM: {vlm:.3f}, OCR: '{text}', Conf: {conf:.3f}")

    save_debug(img, masked, req.prompt, vlm, conf, text, timestep=req.timestep)

    return {"vlm_score": vlm, "ocr_text": text, "ocr_score": conf}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to Qwen-VL model")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    global qwen, ocr

    print(f"Loading Qwen from {args.model} on {args.device}...")
    qwen = QwenScorer(args.model, args.device)

    print("Loading OCR...")
    ocr = OCRScorer()

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
