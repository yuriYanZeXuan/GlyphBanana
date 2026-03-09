"""Async VLM scorer client (alternative implementation)."""

import asyncio
import base64
import io
from typing import List

import aiohttp
from PIL import Image


class VLMScorer:
    """Async VLM scorer using external API."""

    def __init__(self, url: str, session: aiohttp.ClientSession):
        self.url = url
        self.session = session

    async def _score_one(self, img: Image.Image, prompt: str) -> float:
        """Score single image."""
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        img_b64 = base64.b64encode(buf.getvalue()).decode()

        async with self.session.post(self.url, json={"image": img_b64, "prompt": prompt}, timeout=30) as resp:
            resp.raise_for_status()
            data = await resp.json()
            return data.get("score", 0.0)

    async def score_batch(self, images: List[Image.Image], prompts: List[str]) -> List[float]:
        """Score batch of images concurrently."""
        if not images:
            return []

        tasks = [self._score_one(img, p) for img, p in zip(images, prompts)]
        return await asyncio.gather(*tasks)


async def main():
    """Example usage."""
    url = "http://localhost:8000/score"

    # Create test image
    img = Image.new("RGB", (100, 100), color="red")
    images = [img, img]
    prompts = ["a red square", "another red square"]

    async with aiohttp.ClientSession() as session:
        scorer = VLMScorer(url, session)
        scores = await scorer.score_batch(images, prompts)
        print(f"Scores: {scores}")


if __name__ == "__main__":
    asyncio.run(main())
