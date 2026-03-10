#!/usr/bin/env python3
"""GlyphBanana unified generation entry."""

import argparse
from pathlib import Path
from typing import Optional

import torch
from PIL import Image

from infer.generation_utils import DEFAULT_KLEIN_MODEL_PATH, load_text_regions_file
from infer.qwen_image_inference import DEFAULT_MODEL_PATH as DEFAULT_QWEN_MODEL_PATH
from infer.qwen_image_inference import QwenGenerationConfig, QwenImageInference
from infer.zimage_inference import DEFAULT_MODEL_PATH, ZImageGenerationConfig, ZImageInference

DEFAULT_MODEL_PATHS = {
    "zimage": DEFAULT_MODEL_PATH,
    "qwen-image": DEFAULT_QWEN_MODEL_PATH,
}


def parse_args():
    parser = argparse.ArgumentParser(description="GlyphBanana 生成")
    parser.add_argument("--backend", choices=["zimage", "qwen-image"], default="zimage")
    parser.add_argument("--prompt", required=True, help="生成提示词（含文字描述）")
    parser.add_argument("--text", nargs="+", required=True, help="待渲染的文本/公式列表")
    parser.add_argument("--output", default="output.png", help="输出路径")
    parser.add_argument("--text-regions-file", help="可选 JSON 文件，传入手工 layout 后跳过 VLM planner")
    parser.add_argument("--model-path", help="模型路径；不传则按 backend 使用默认路径")
    parser.add_argument("--device", default="cuda", help="设备")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--steps", type=int, default=20, help="推理步数")
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--no-harmonize", action="store_true", help="跳过 Pass 3 风格化")
    parser.add_argument("--klein-model-path", default=DEFAULT_KLEIN_MODEL_PATH)
    parser.add_argument("--klein-steps", type=int, default=10)
    parser.add_argument("--klein-guidance", type=float, default=4.0)
    parser.add_argument("--true-cfg-scale", type=float, default=4.0, help="qwen-image 专用 true CFG")
    parser.add_argument("--guidance-scale", type=float, help="qwen-image guidance scale，可选")
    return parser.parse_args()


def build_backend(
    backend: str = "zimage",
    model_path: Optional[str] = None,
    device: str = "cuda",
    dtype=torch.bfloat16,
):
    """构建可复用生成后端。"""
    if backend == "zimage":
        return ZImageInference(model_path or DEFAULT_MODEL_PATHS["zimage"], device=device, dtype=dtype)
    if backend == "qwen-image":
        return QwenImageInference(model_path=model_path or DEFAULT_MODEL_PATHS["qwen-image"], device=device, dtype=dtype)
    raise ValueError(f"Unsupported backend: {backend}")


def build_generation_config(
    backend: str,
    *,
    seed: int,
    steps: int,
    height: int,
    width: int,
    no_harmonize: bool,
    klein_model_path: str,
    klein_steps: int,
    klein_guidance: float,
    true_cfg_scale: float,
    guidance_scale: Optional[float],
):
    common = dict(
        height=height,
        width=width,
        seed=seed,
        use_harmonization=not no_harmonize,
        klein_model_path=klein_model_path,
        klein_steps=klein_steps,
    )
    if backend == "zimage":
        return ZImageGenerationConfig(
            num_inference_steps=steps,
            klein_guidance_scale=klein_guidance,
            **common,
        )
    if backend == "qwen-image":
        return QwenGenerationConfig(
            num_inference_steps=steps,
            true_cfg_scale=true_cfg_scale,
            guidance_scale=guidance_scale,
            klein_guidance_scale=klein_guidance,
            **common,
        )
    raise ValueError(f"Unsupported backend: {backend}")


def generate_image(
    prompt: str,
    text: list[str] | str,
    output: Optional[str] = None,
    backend: str = "zimage",
    text_regions: Optional[list[dict]] = None,
    model_path: Optional[str] = None,
    device: str = "cuda",
    seed: int = 42,
    steps: int = 20,
    height: int = 1024,
    width: int = 1024,
    no_harmonize: bool = False,
    klein_model_path: str = DEFAULT_KLEIN_MODEL_PATH,
    klein_steps: int = 10,
    klein_guidance: float = 4.0,
    true_cfg_scale: float = 4.0,
    guidance_scale: Optional[float] = None,
) -> Image.Image:
    """包级快捷入口。"""
    texts = [text] if isinstance(text, str) else list(text)
    runner = build_backend(backend=backend, model_path=model_path, device=device)
    config = build_generation_config(
        backend,
        seed=seed,
        steps=steps,
        height=height,
        width=width,
        no_harmonize=no_harmonize,
        klein_model_path=klein_model_path,
        klein_steps=klein_steps,
        klein_guidance=klein_guidance,
        true_cfg_scale=true_cfg_scale,
        guidance_scale=guidance_scale,
    )
    image = runner.generate(prompt=prompt, text_contents=texts, text_regions=text_regions, config=config, output_path=output)

    if output:
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        image.convert("RGB").save(output_path)
        print(f"Saved to: {output_path}")
    return image


def main():
    args = parse_args()
    text_regions = load_text_regions_file(args.text_regions_file)
    generate_image(
        prompt=args.prompt,
        text=args.text,
        output=args.output,
        backend=args.backend,
        text_regions=text_regions,
        model_path=args.model_path or DEFAULT_MODEL_PATHS[args.backend],
        device=args.device,
        seed=args.seed,
        steps=args.steps,
        height=args.height,
        width=args.width,
        no_harmonize=args.no_harmonize,
        klein_model_path=args.klein_model_path,
        klein_steps=args.klein_steps,
        klein_guidance=args.klein_guidance,
        true_cfg_scale=args.true_cfg_scale,
        guidance_scale=args.guidance_scale,
    )


if __name__ == "__main__":
    main()
