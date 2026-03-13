#!/usr/bin/env python3
"""
GlyphBanana Lightweight Generation Script

Three-stage inference:
  Pass 1: Generate reference image with full prompt
  VLM:    Analyze reference image, autonomously plan text typography
  Pass 2: Generate text image from same noise with clean prompt + glyph injection
  Pass 3: Style harmonization (optional)

Usage:
  python generate.py --prompt "poster text" --text "content"
  python generate.py --prompt "..." --text "..." --no-harmonize  # Skip stylization
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
    parser = argparse.ArgumentParser(description="GlyphBanana Generation")
    parser.add_argument("--prompt", required=True, help="Generation prompt (with text description)")
    parser.add_argument("--text", nargs="+", required=True, help="List of texts/formulas to render")
    parser.add_argument("--output", default="output.png", help="Output path")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH, help="Model path")
    parser.add_argument("--device", default="cuda", help="Device")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--steps", type=int, default=20, help="Inference steps")
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--no-harmonize", action="store_true", help="Skip Pass 3 stylization")
    parser.add_argument("--klein-model-path", default="/mnt/tidalfs-bdsz01/usr/tusen/yanzexuan/weight/flux2-klein")
    parser.add_argument("--klein-steps", type=int, default=10)
    parser.add_argument("--klein-guidance", type=float, default=4.0)
    return parser.parse_args()


def load_pipeline(model_path: str, device: str, dtype=torch.bfloat16):
    """Load ZImagePipeline"""
    print(f"Loading model from {model_path}...")
    pipe = ZImagePipeline.from_pretrained(model_path, torch_dtype=dtype, low_cpu_mem_usage=False)
    pipe.to(device)
    print("Model loaded.")
    return pipe


def prepare_noise(pipeline, height: int, width: int, device: str, generator=None):
    """Prepare shared noise"""
    latent_h = 2 * (height // (pipeline.vae_scale_factor * 2))
    latent_w = 2 * (width // (pipeline.vae_scale_factor * 2))
    channels = pipeline.transformer.in_channels
    return torch.randn((1, channels, latent_h, latent_w), generator=generator, device=device, dtype=torch.float32)


def decode_latent(pipeline, latent: torch.Tensor) -> Image.Image:
    """Decode latent to PIL Image"""
    latent = latent.to(pipeline.vae.dtype)
    latent = (latent / pipeline.vae.config.scaling_factor) + pipeline.vae.config.shift_factor
    with torch.no_grad():
        image = pipeline.vae.decode(latent, return_dict=False)[0]
    return pipeline.image_processor.postprocess(image, output_type="pil")[0]


def run_pass1_reference(pipeline, prompt: str, noise: torch.Tensor, timesteps, device: str) -> Image.Image:
    """Pass 1: Generate reference image with full prompt"""
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
    """Pass 2: Denoise with clean prompt + glyph injection"""
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

        # Glyph injection
        if "glyph_injector" in injection_data:
            injector = injection_data["glyph_injector"]
            latent = injector.inject_latent(latent, injection_data, step_idx + 1)

    return latent


def pixel_composite_text(background: Image.Image, injection_data: dict) -> Image.Image:
    """Pixel-space composition: paste text stroke pixels using mask"""
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
    """Pass 3: FluxKlein stylization - multi-variant generation + VLM selection"""
    klein_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "baselines", "fluxklein")
    if klein_dir not in sys.path:
        sys.path.insert(0, klein_dir)
    from inference_fluxklein import FluxKleinGenerator

    print(f"Loading FluxKlein from {klein_model_path}...")
    klein = FluxKleinGenerator(model_path=klein_model_path, device=device)

    # Generate style prompt
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
        """Blend result with base_img using mask"""
        return Image.fromarray((
            mask_3ch * np.array(result_img).astype(np.float32)
            + (1 - mask_3ch) * np.array(base_img).astype(np.float32)
        ).clip(0, 255).astype(np.uint8))

    variants = []

    # Variant 1: Single image condition + mask blending (seed)
    print("  [1/3] Generating variant 1 (single image + mask)...")
    result1 = klein.pipe(
        prompt=style_prompt, image=[pass2_image],
        num_inference_steps=steps, guidance_scale=guidance,
        generator=_make_generator(0),
    ).images[0]
    variants.append(("klein_single_masked", _apply_mask(result1, pass2_image)))

    # Variant 2: Single image condition, no mask (seed+1)
    print("  [2/3] Generating variant 2 (single image, no mask)...")
    result2 = klein.pipe(
        prompt=style_prompt, image=[pass2_image],
        num_inference_steps=steps, guidance_scale=guidance,
        generator=_make_generator(1),
    ).images[0]
    variants.append(("klein_nomask", result2.convert("RGB") if result2.mode != "RGB" else result2))

    # Variant 3: Dual image condition (pass2 + glyph template), no mask (seed+2)
    print("  [3/3] Generating variant 3 (dual image)...")
    result3 = klein.pipe(
        prompt=style_prompt, image=[pass2_image, template],
        num_inference_steps=steps, guidance_scale=guidance,
        generator=_make_generator(2),
    ).images[0]
    variants.append(("klein_dual", result3.convert("RGB") if result3.mode != "RGB" else result3))

    # Save concatenated image
    if output_path:
        concat_path = output_path.replace(".png", "_pass3_variants.png")
        w, h = pass2_image.size
        concat = Image.new("RGB", (w * len(variants), h))
        for i, (name, img) in enumerate(variants):
            concat.paste(img.resize((w, h), Image.LANCZOS), (w * i, 0))
        concat.save(concat_path)
        print(f"  Saved variants concat to: {concat_path}")

    # VLM selection
    if vlm_agent:
        print("  VLM selecting best variant...")
        images = [img for _, img in variants]
        best_idx = vlm_agent.select_best_image(images, prompt)
        print(f"  Selected: {variants[best_idx][0]} (index {best_idx})")
        return variants[best_idx][1]
    
    # Default return first
    return variants[0][1]


def main():
    args = parse_args()

    # Load model
    device = args.device
    generator = torch.Generator(device=device).manual_seed(args.seed)
    pipeline = load_pipeline(args.model_path, device)

    # Initialize VLM Agent and Glyph Injector
    print("Initializing VLM Agent...")
    vlm_agent = VLMAgent()
    print("Initializing Glyph Injector...")
    glyph_injector = create_glyph_injector(pipeline, device=device)

    # Add deterministic suffix to optimize prompt
    working_prompt = args.prompt + ",horizontal text layout."
    print(f"Working prompt: {working_prompt}")

    # Prepare shared noise
    noise = prepare_noise(pipeline, args.height, args.width, device, generator)
    pipeline.scheduler.set_timesteps(args.steps, device=device)
    timesteps = pipeline.scheduler.timesteps

    # === Pass 1: Generate reference image ===
    print("\n=== Pass 1: Reference Image ===")
    reference_image = run_pass1_reference(pipeline, working_prompt, noise.clone(), timesteps, device)

    # === VLM typography planning ===
    print("\n=== VLM Typography Planning ===")
    typography_plan = vlm_agent.analyze_typography(reference_image, working_prompt, args.text)

    # === Pass 2: Clean background + glyph injection ===
    print("\n=== Pass 2: Glyph Injection ===")
    pipeline.scheduler.set_timesteps(args.steps, device=device)
    timesteps = pipeline.scheduler.timesteps

    clean_prompt = vlm_agent.generate_clean_prompt(working_prompt, typography_plan)
    print(f"Clean prompt: {clean_prompt}")

    image_size = (args.width, args.height)
    injection_data = glyph_injector.prepare_injection_from_plan(typography_plan, image_size, noise, timesteps)

    # Check if mask is valid
    if injection_data["full_mask"].max() == 0:
        print("Warning: Empty glyph mask, returning reference image")
        reference_image.save(args.output)
        return

    # Inject and generate background
    injection_data["glyph_injector"] = glyph_injector
    background_latent = run_pass2_injection(pipeline, clean_prompt, noise, timesteps, injection_data, device)
    background = decode_latent(pipeline, background_latent)

    # Pixel-space composition
    pass2_image = pixel_composite_text(background, injection_data)
    print("Pass 2 completed.")

    # === Pass 3: Stylization (optional) ===
    if not args.no_harmonize:
        print("\n=== Pass 3: Harmonization ===")
        # Unload main model to free VRAM
        pipeline.to("cpu")
        torch.cuda.empty_cache()

        final_image = run_pass3_klein(
            pass2_image, injection_data, args.klein_model_path,
            working_prompt, args.klein_steps, args.klein_guidance, args.seed, device,
            vlm_agent=vlm_agent, output_path=args.output
        )

        # Restore main model
        pipeline.to(device)
    else:
        final_image = pass2_image

    # Save result
    if final_image.mode != "RGB":
        final_image = final_image.convert("RGB")
    final_image.save(args.output)
    print(f"\nSaved to: {args.output}")


if __name__ == "__main__":
    main()
