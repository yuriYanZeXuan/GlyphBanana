"""Shared generation helpers for GlyphBanana backends."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image

from infer.VLM_agent import VLMAgent

DEFAULT_KLEIN_MODEL_PATH = "/mnt/tidalfs-bdsz01/usr/tusen/yanzexuan/weight/flux2-klein"
_TEXT_REGION_KEYS = (
    "content",
    "bbox",
    "font",
    "font_path",
    "font_weight",
    "font_size_ratio",
    "color",
    "is_latex",
    "alignment",
    "rotation",
)
_TEXT_REGION_DEFAULTS = {
    "content": "",
    "bbox": [0, 0, 1, 1],
    "font": "auto",
    "font_path": None,
    "font_weight": "regular",
    "font_size_ratio": 0.7,
    "color": "#FFFFFF",
    "is_latex": False,
    "alignment": "center",
    "rotation": 0,
}


def text_regions_to_plan(text_regions: list[dict]) -> dict:
    """Convert manual text regions to a unified typography plan."""
    plan_regions = [
        {key: region.get(key, _TEXT_REGION_DEFAULTS[key]) for key in _TEXT_REGION_KEYS}
        for region in text_regions
    ]
    return {
        "image_analysis": {
            "background_style": "manual layout",
            "dominant_colors": ["#000000"],
            "text_style_hint": "manual layout override",
        },
        "text_regions": plan_regions,
    }


def load_text_regions_file(path: Optional[str]) -> Optional[list[dict]]:
    if not path:
        return None
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("text_regions", "regions"):
            value = data.get(key)
            if isinstance(value, list):
                return value
    raise ValueError(f"Unsupported text regions schema: {path}")


def run_pass3_klein(
    pass2_image: Image.Image,
    injection_data: dict,
    klein_model_path: str,
    prompt: str,
    steps: int,
    guidance: float,
    seed: int,
    device: str,
    vlm_agent: Optional[VLMAgent] = None,
    output_path: Optional[str] = None,
) -> Image.Image:
    """Run FluxKlein harmonization and optionally save variant concat."""
    project_root = Path(__file__).resolve().parent.parent
    klein_dir = project_root / "baselines" / "fluxklein"
    if str(klein_dir) not in sys.path:
        sys.path.insert(0, str(klein_dir))
    from inference_fluxklein import FluxKleinGenerator

    print(f"Loading FluxKlein from {klein_model_path}...")
    klein = FluxKleinGenerator(model_path=klein_model_path, device=device)

    style_prompt = f"Make text harmonize with the {prompt.split()[0]} style background, matching color and texture."
    print(f"Style prompt: {style_prompt}")

    template = injection_data["combined_template"]
    if template.size != pass2_image.size:
        template = template.resize(pass2_image.size, Image.LANCZOS)

    binary_mask = injection_data["full_mask"]
    mask = (binary_mask > 127).astype(np.float32)
    if mask.shape[:2] != (pass2_image.height, pass2_image.width):
        resized = Image.fromarray((mask * 255).astype(np.uint8)).resize(pass2_image.size, Image.NEAREST)
        mask = np.array(resized).astype(np.float32) / 255.0
    mask_3ch = mask[:, :, np.newaxis]

    def _make_generator(seed_offset=0):
        return torch.Generator(device=device).manual_seed(seed + seed_offset)

    def _ensure_rgb(image: Image.Image) -> Image.Image:
        return image.convert("RGB") if image.mode != "RGB" else image

    def _apply_mask(result_img, base_img):
        return Image.fromarray(
            (
                mask_3ch * np.array(result_img).astype(np.float32)
                + (1 - mask_3ch) * np.array(base_img).astype(np.float32)
            ).clip(0, 255).astype(np.uint8)
        )

    variants = []

    print("  [1/3] Generating variant 1 (single image + mask)...")
    result1 = klein.pipe(
        prompt=style_prompt,
        image=[pass2_image],
        num_inference_steps=steps,
        guidance_scale=guidance,
        generator=_make_generator(0),
    ).images[0]
    variants.append(("klein_single_masked", _apply_mask(_ensure_rgb(result1), pass2_image)))

    print("  [2/3] Generating variant 2 (single image, no mask)...")
    result2 = klein.pipe(
        prompt=style_prompt,
        image=[pass2_image],
        num_inference_steps=steps,
        guidance_scale=guidance,
        generator=_make_generator(1),
    ).images[0]
    variants.append(("klein_nomask", _ensure_rgb(result2)))

    print("  [3/3] Generating variant 3 (dual image)...")
    result3 = klein.pipe(
        prompt=style_prompt,
        image=[pass2_image, template],
        num_inference_steps=steps,
        guidance_scale=guidance,
        generator=_make_generator(2),
    ).images[0]
    variants.append(("klein_dual", _ensure_rgb(result3)))

    if output_path:
        stem = str(Path(output_path).with_suffix(""))
        concat_path = f"{stem}_pass3_variants.png"
        w, h = pass2_image.size
        concat = Image.new("RGB", (w * len(variants), h))
        for i, (_, img) in enumerate(variants):
            concat.paste(_ensure_rgb(img).resize((w, h), Image.LANCZOS), (w * i, 0))
        concat.save(concat_path)
        print(f"Saved variants concat to: {concat_path}")

    if vlm_agent:
        images = [img for _, img in variants]
        best_idx = vlm_agent.select_best_image(images, prompt)
        print(f"Selected: {variants[best_idx][0]} (index {best_idx})")
        return variants[best_idx][1]

    return variants[0][1]
