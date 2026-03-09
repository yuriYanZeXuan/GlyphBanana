"""Reward client for fetching scores from reward servers."""

import base64
import random
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO

import requests
from PIL import Image


def encode_image(img: Image.Image) -> str:
    """Encode PIL image to base64 string."""
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def fetch_reward(url: str, img_b64: str, prompt: str, mask_b64: str | None = None,text_gt: str = "") -> dict:
    """Fetch reward from a single server."""
    payload = {"image": img_b64, "prompt": prompt}
    if mask_b64:
        payload["mask"] = mask_b64
    if text_gt:
        payload["text_gt"]=text_gt

    resp = requests.post(url, json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()

    if isinstance(data, list):
        data = data[0] if data else {}

    return {
        "vlm": float(data.get("vlm_score", 0.0)),
        "ocr": float(data.get("ocr_score", 0.0)),
        "ocr_text": data.get("ocr_text", ""),
    }


class RewardClient:
    """Client for fetching rewards from multiple servers with load balancing."""

    def __init__(self, urls: list[str], ocr_w: float = 0.7, vlm_w: float = 0.3):
        if not urls:
            raise ValueError("Server URLs cannot be empty")
        self.urls = urls
        self.ocr_w = ocr_w
        self.vlm_w = vlm_w
        self.executor = ThreadPoolExecutor(max_workers=len(urls))
        self._idx = random.randint(0, len(urls) - 1)

    def score(self, images: list[Image.Image], prompts: list[str], masks: list[Image.Image | None] | None = None,text_gts: list[str] = []) -> list[dict]:
        """Get rewards for a batch of images."""
        masks = masks or [None] * len(images)
        text_gts = text_gts or [""] * len(images)
        futures = []
        for img, prompt, mask,text_gt in zip(images, prompts, masks,text_gts):
            url = self.urls[self._idx % len(self.urls)]
            self._idx += 1

            img_b64 = encode_image(img)
            mask_b64 = encode_image(mask) if mask else None

            futures.append(self.executor.submit(fetch_reward, url, img_b64, prompt, mask_b64,text_gt))

        results = []
        for f in futures:
            r = f.result()
            r["combined_score"] = r["vlm"] * self.vlm_w + r["ocr"] * self.ocr_w
            results.append(r)

        return results
