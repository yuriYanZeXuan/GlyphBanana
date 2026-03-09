"""Pseudo reward server returning random scores for testing."""

import argparse
import base64
import io
import random
from contextlib import asynccontextmanager

import numpy as np
import uvicorn
from fastapi import FastAPI
from PIL import Image
from pydantic import BaseModel

from debug_utils import save_debug


class ScoreRequest(BaseModel):
    image: str
    prompt: str
    mask: str | None = None
    timestep: str | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Pseudo reward server starting...")
    yield
    print("Pseudo reward server shutting down...")


app = FastAPI(lifespan=lifespan)


@app.post("/score")
async def score(req: ScoreRequest):
    # Decode inputs
    img = Image.open(io.BytesIO(base64.b64decode(req.image))).convert("RGB")

    mask = None
    if req.mask:
        mask = Image.open(io.BytesIO(base64.b64decode(req.mask))).convert("L")
        if mask.size != img.size:
            mask = mask.resize(img.size, Image.NEAREST)

    masked = img
    if mask:
        marr = (np.array(mask) > 127).astype(np.uint8)
        iarr = np.array(img)
        iarr[marr == 0] = 255
        masked = Image.fromarray(iarr)

    # Random scores
    vlm = random.random()
    ocr = random.random()

    save_debug(img, masked, req.prompt, vlm, ocr, "")

    return {"vlm_score": vlm, "ocr_text": "", "ocr_score": ocr}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
