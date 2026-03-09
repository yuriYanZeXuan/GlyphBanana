#!/usr/bin/env python3
"""
GlyphBanana 精简生成脚本

三阶段推理:
  Pass 1: 用完整 prompt 生成参考图
  VLM:    分析参考图，自主规划文本排版
  Pass 2: 从相同噪声用 clean prompt + 字形注入生成文字图
  Pass 3: 风格化 harmonization (可选)

用法:
  python generate.py --prompt "海报文字" --text "内容"
  python generate.py --prompt "..." --text "..." --no-harmonize  # 跳过风格化
"""

import os
import sys
import argparse
from typing import Optional

os.environ["TORCH_COMPILE_DISABLE"] = "1"

import torch
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from train.zimage_ip.pipeline_z_image import ZImagePipeline
from infer.VLM_agent import VLMAgent, _add_grid_overlay
from infer.glyph_injector import GlyphInjector, create_glyph_injector

DEFAULT_MODEL_PATH = "/mnt/tidalfs-bdsz01/usr/tusen/yanzexuan/weight/Z-Image"


def parse_args():
    parser = argparse.ArgumentParser(description="GlyphBanana 生成")
    parser.add_argument("--prompt", required=True, help="生成提示词（含文字描述）")
    parser.add_argument("--text", nargs="+", required=True, help="待渲染的文本/公式列表")
    parser.add_argument("--output", default="output.png", help="输出路径")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH, help="模型路径")
    parser.add_argument("--device", default="cuda", help="设备")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--steps", type=int, default=20, help="推理步数")
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--no-harmonize", action="store_true", help="跳过 Pass 3 风格化")
    parser.add_argument("--klein-model-path", default="/mnt/tidalfs-bdsz01/usr/tusen/yanzexuan/weight/flux2-klein")
    parser.add_argument("--klein-steps", type=int, default=10)
    parser.add_argument("--klein-guidance", type=float, default=4.0)
    return parser.parse_args()


def load_pipeline(model_path: str, device: str, dtype=torch.bfloat16):
    """加载 ZImagePipeline"""
    print(f"Loading model from {model_path}...")
    pipe = ZImagePipeline.from_pretrained(model_path, torch_dtype=dtype, low_cpu_mem_usage=False)
    pipe.to(device)
    print("Model loaded.")
    return pipe


def prepare_noise(pipeline, height: int, width: int, device: str, generator=None):
    """准备共享噪声"""
    latent_h = 2 * (height // (pipeline.vae_scale_factor * 2))
    latent_w = 2 * (width // (pipeline.vae_scale_factor * 2))
    channels = pipeline.transformer.in_channels
    return torch.randn((1, channels, latent_h, latent_w), generator=generator, device=device, dtype=torch.float32)


def decode_latent(pipeline, latent: torch.Tensor) -> Image.Image:
    """解码 latent 为 PIL Image"""
    latent = latent.to(pipeline.vae.dtype)
    latent = (latent / pipeline.vae.config.scaling_factor) + pipeline.vae.config.shift_factor
    with torch.no_grad():
        image = pipeline.vae.decode(latent, return_dict=False)[0]
    return pipeline.image_processor.postprocess(image, output_type="pil")[0]


def run_pass1_reference(pipeline, prompt: str, noise: torch.Tensor, timesteps, device: str) -> Image.Image:
    """Pass 1: 用完整 prompt 生成参考图"""
    dtype = pipeline.transformer.dtype
    latent = noise.clone()
    prompt_embeds, _ = pipeline.encode_prompt(prompt=prompt, device=device, do_classifier_free_guidance=False)

    for t in timesteps:
        timestep = t.expand(1)
        timestep_norm = (1000 - timestep) / 1000
        latent_input = latent.to(dtype).unsqueeze(2)
        latent_list = list(latent_input.unbind(dim=0))

        with torch.no_grad():
            model_out = pipeline.transformer(latent_list, timestep_norm, prompt_embeds, return_dict=False)[0]

        noise_pred = -torch.stack([o.float() for o in model_out], dim=0).squeeze(2)
        latent = pipeline.scheduler.step(noise_pred.to(torch.float32), t, latent, return_dict=False)[0]

    return decode_latent(pipeline, latent)


def run_pass2_injection(
    pipeline, prompt: str, noise: torch.Tensor, timesteps, injection_data: dict, device: str
) -> torch.Tensor:
    """Pass 2: 用 clean prompt 去噪 + 字形注入"""
    dtype = pipeline.transformer.dtype
    latent = noise.clone()
    prompt_embeds, _ = pipeline.encode_prompt(prompt=prompt, device=device, do_classifier_free_guidance=False)

    for step_idx, t in enumerate(timesteps):
        timestep = t.expand(1)
        timestep_norm = (1000 - timestep) / 1000
        latent_input = latent.to(dtype).unsqueeze(2)
        latent_list = list(latent_input.unbind(dim=0))

        with torch.no_grad():
            model_out = pipeline.transformer(latent_list, timestep_norm, prompt_embeds, return_dict=False)[0]

        noise_pred = -torch.stack([o.float() for o in model_out], dim=0).squeeze(2)
        latent = pipeline.scheduler.step(noise_pred.to(torch.float32), t, latent, return_dict=False)[0]

        # 字形注入
        if "glyph_injector" in injection_data:
            injector = injection_data["glyph_injector"]
            latent = injector.inject_latent(latent, injection_data, step_idx + 1)

    return latent


def pixel_composite_text(background: Image.Image, injection_data: dict) -> Image.Image:
    """像素空间合成：用 mask 只贴文字笔画像素"""
    template = injection_data["combined_template"]
    mask = injection_data["full_mask"]

    bg_arr = np.array(background)
    tpl_arr = np.array(template)

    if tpl_arr.shape[:2] != bg_arr.shape[:2]:
        template = template.resize(background.size, Image.LANCZOS)
        tpl_arr = np.array(template)
    if mask.shape[:2] != bg_arr.shape[:2]:
        mask = np.array(Image.fromarray(mask).resize(background.size, Image.NEAREST))

    stroke_mask = mask > 127
    result = bg_arr.copy()
    result[stroke_mask] = tpl_arr[stroke_mask]

    return Image.fromarray(result)


def run_pass3_klein(
    pass2_image: Image.Image, injection_data: dict, klein_model_path: str,
    prompt: str, steps: int, guidance: float, seed: int, device: str,
    vlm_agent=None, output_path: str = None,
) -> Image.Image:
    """Pass 3: FluxKlein 风格化 - 多 variant 生成 + VLM 选优"""
    klein_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "baselines", "fluxklein")
    if klein_dir not in sys.path:
        sys.path.insert(0, klein_dir)
    from inference_fluxklein import FluxKleinGenerator

    print(f"Loading FluxKlein from {klein_model_path}...")
    klein = FluxKleinGenerator(model_path=klein_model_path, device=device)

    # 生成风格化提示词
    style_prompt = f"Make text harmonize with the {prompt.split()[0]} style background, matching color and texture."
    print(f"Style prompt: {style_prompt}")

    template = injection_data["combined_template"]
    if template.size != pass2_image.size:
        template = template.resize(pass2_image.size, Image.LANCZOS)

    binary_mask = injection_data["full_mask"]
    mask = (binary_mask > 127).astype(np.float32)
    if mask.shape[:2] != (pass2_image.height, pass2_image.width):
        mask = np.array(Image.fromarray((mask * 255).astype(np.uint8)).resize(pass2_image.size, Image.NEAREST)).astype(np.float32) / 255.0
    mask_3ch = mask[:, :, np.newaxis]

    def _make_generator(seed_offset=0):
        return torch.Generator(device=device).manual_seed(seed + seed_offset)

    def _apply_mask(result_img, base_img):
        """将结果与 base_img 按 mask 混合"""
        return Image.fromarray((
            mask_3ch * np.array(result_img).astype(np.float32)
            + (1 - mask_3ch) * np.array(base_img).astype(np.float32)
        ).clip(0, 255).astype(np.uint8))

    variants = []

    # Variant 1: 单图条件 + mask 混合 (seed)
    print("  [1/3] Generating variant 1 (single image + mask)...")
    result1 = klein.pipe(
        prompt=style_prompt, image=[pass2_image],
        num_inference_steps=steps, guidance_scale=guidance,
        generator=_make_generator(0),
    ).images[0]
    variants.append(("klein_single_masked", _apply_mask(result1, pass2_image)))

    # Variant 2: 单图条件，无 mask (seed+1)
    print("  [2/3] Generating variant 2 (single image, no mask)...")
    result2 = klein.pipe(
        prompt=style_prompt, image=[pass2_image],
        num_inference_steps=steps, guidance_scale=guidance,
        generator=_make_generator(1),
    ).images[0]
    variants.append(("klein_nomask", result2.convert("RGB") if result2.mode != "RGB" else result2))

    # Variant 3: 双图条件（pass2 + glyph 模板），无 mask (seed+2)
    print("  [3/3] Generating variant 3 (dual image)...")
    result3 = klein.pipe(
        prompt=style_prompt, image=[pass2_image, template],
        num_inference_steps=steps, guidance_scale=guidance,
        generator=_make_generator(2),
    ).images[0]
    variants.append(("klein_dual", result3.convert("RGB") if result3.mode != "RGB" else result3))

    # 保存拼接图
    if output_path:
        concat_path = output_path.replace(".png", "_pass3_variants.png")
        w, h = pass2_image.size
        concat = Image.new("RGB", (w * len(variants), h))
        for i, (name, img) in enumerate(variants):
            concat.paste(img.resize((w, h), Image.LANCZOS), (w * i, 0))
        concat.save(concat_path)
        print(f"  Saved variants concat to: {concat_path}")

    # VLM 选优
    if vlm_agent:
        print("  VLM selecting best variant...")
        images = [img for _, img in variants]
        best_idx = vlm_agent.select_best_image(images, prompt)
        print(f"  Selected: {variants[best_idx][0]} (index {best_idx})")
        return variants[best_idx][1]
    
    # 默认返回第一个
    return variants[0][1]


def main():
    args = parse_args()

    # 加载模型
    device = args.device
    generator = torch.Generator(device=device).manual_seed(args.seed)
    pipeline = load_pipeline(args.model_path, device)

    # 初始化 VLM Agent 和 Glyph Injector
    print("Initializing VLM Agent...")
    vlm_agent = VLMAgent()
    print("Initializing Glyph Injector...")
    glyph_injector = create_glyph_injector(pipeline, device=device)

    # 添加 deterministic 后缀优化 prompt
    working_prompt = args.prompt + ",horizontal text layout."
    print(f"Working prompt: {working_prompt}")

    # 准备共享噪声
    noise = prepare_noise(pipeline, args.height, args.width, device, generator)
    pipeline.scheduler.set_timesteps(args.steps, device=device)
    timesteps = pipeline.scheduler.timesteps

    # === Pass 1: 生成参考图 ===
    print("\n=== Pass 1: Reference Image ===")
    reference_image = run_pass1_reference(pipeline, working_prompt, noise.clone(), timesteps, device)

    # === VLM 排版规划 ===
    print("\n=== VLM Typography Planning ===")
    typography_plan = vlm_agent.analyze_typography(reference_image, working_prompt, args.text)

    # === Pass 2: Clean 背景 + 字形注入 ===
    print("\n=== Pass 2: Glyph Injection ===")
    pipeline.scheduler.set_timesteps(args.steps, device=device)
    timesteps = pipeline.scheduler.timesteps

    clean_prompt = vlm_agent.generate_clean_prompt(working_prompt, typography_plan)
    print(f"Clean prompt: {clean_prompt}")

    image_size = (args.width, args.height)
    injection_data = glyph_injector.prepare_injection_from_plan(typography_plan, image_size, noise, timesteps)

    # 检查 mask 是否有效
    if injection_data["full_mask"].max() == 0:
        print("Warning: Empty glyph mask, returning reference image")
        reference_image.save(args.output)
        return

    # 注入生成背景
    injection_data["glyph_injector"] = glyph_injector
    background_latent = run_pass2_injection(pipeline, clean_prompt, noise, timesteps, injection_data, device)
    background = decode_latent(pipeline, background_latent)

    # 像素空间合成
    pass2_image = pixel_composite_text(background, injection_data)
    print("Pass 2 completed.")

    # === Pass 3: 风格化 (可选) ===
    if not args.no_harmonize:
        print("\n=== Pass 3: Harmonization ===")
        # 卸载主模型释放显存
        pipeline.to("cpu")
        torch.cuda.empty_cache()

        final_image = run_pass3_klein(
            pass2_image, injection_data, args.klein_model_path,
            working_prompt, args.klein_steps, args.klein_guidance, args.seed, device,
            vlm_agent=vlm_agent, output_path=args.output
        )

        # 恢复主模型
        pipeline.to(device)
    else:
        final_image = pass2_image

    # 保存结果
    if final_image.mode != "RGB":
        final_image = final_image.convert("RGB")
    final_image.save(args.output)
    print(f"\nSaved to: {args.output}")


if __name__ == "__main__":
    main()
