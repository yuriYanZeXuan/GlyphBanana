"""Debug utilities for saving reward evaluation samples."""

import os
import uuid
from datetime import datetime
from pathlib import Path

from PIL import Image

SAVE_INTERVAL = int(os.environ.get("REWARD_DEBUG_INTERVAL", "100"))
_save_counter = 0


def save_debug(
    image: Image.Image,
    masked: Image.Image,
    prompt: str,
    vlm: float,
    ocr: float,
    ocr_text: str = "",
    prefix: str = "reward",
    timestep: str | None = None,
) -> None:
    """Save debug artifacts. Called every SAVE_INTERVAL times."""
    global _save_counter
    _save_counter += 1

    if SAVE_INTERVAL > 0 and _save_counter % SAVE_INTERVAL != 0:
        return

    out_dir = Path(os.environ.get("REWARD_DEBUG_DIR", "reward_debug"))
    out_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    uid = uuid.uuid4().hex[:8]
    parts = [prefix, ts, uid]
    if timestep:
        parts.append(f"t_{timestep.replace(':', '-').replace(' ', '_')}")
    parts.extend([f"vlm_{vlm:.2f}", f"ocr_{ocr:.2f}"])

    base = "_".join(parts)

    image.save(out_dir / f"{base}.png")
    masked.save(out_dir / f"{base}_masked.png")

    lines = [f"Prompt: {prompt}", f"VLM: {vlm}", f"OCR: {ocr}"]
    if timestep:
        lines.append(f"Timestep: {timestep}")
    if ocr_text:
        lines.append(f"OCR Text: {ocr_text}")

    (out_dir / f"{base}.txt").write_text("\n".join(lines), encoding="utf-8")
