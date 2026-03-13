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

from models.zimage_ip.pipeline_z_image import ZImagePipeline
from models.qwen_ip.pipeline_qwenimage import (
    calculate_shift as calculate_qwen_shift,
    retrieve_timesteps as retrieve_qwen_timesteps,
)
from infer.VLM_agent import VLMAgent, _add_grid_overlay
from infer.glyph_injector import GlyphInjector, create_glyph_injector

DEFAULT_ZIMAGE_MODEL_PATH = "/mnt/tidalfs-bdsz01/usr/tusen/yanzexuan/weight/Z-Image"
DEFAULT_QWEN_MODEL_PATH = "/mnt/tidalfs-bdsz01/usr/tusen/yanzexuan/weight/qwen-image-2512"


def parse_args():
    parser = argparse.ArgumentParser(description="GlyphBanana Generation")
    parser.add_argument("--backend", choices=["zimage", "qwen"], default="zimage", help="Inference backend")
    parser.add_argument("--prompt", required=True, help="Generation prompt (with text description)")
    parser.add_argument("--text", nargs="+", required=True, help="List of texts/formulas to render")
    parser.add_argument("--output", default="output.png", help="Output path")
    parser.add_argument("--model-path", default=None, help="Model path")
    parser.add_argument("--device", default="cuda", help="Device")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--steps", type=int, default=20, help="Inference steps")
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--qwen-true-cfg-scale", type=float, default=4.0, help="Qwen true CFG scale")
    parser.add_argument("--qwen-guidance-scale", type=float, default=None, help="Qwen distilled guidance scale")
    parser.add_argument("--no-harmonize", action="store_true", help="Skip Pass 3 stylization")
    parser.add_argument("--klein-model-path", default="/mnt/tidalfs-bdsz01/usr/tusen/yanzexuan/weight/flux2-klein")
    parser.add_argument("--klein-steps", type=int, default=10)
    parser.add_argument("--klein-guidance", type=float, default=4.0)
    return parser.parse_args()


def resolve_model_path(backend: str, model_path: Optional[str]) -> str:
    if model_path:
        return model_path
    if backend == "qwen":
        return DEFAULT_QWEN_MODEL_PATH
    return DEFAULT_ZIMAGE_MODEL_PATH


def load_zimage_pipeline(model_path: str, device: str, dtype=torch.bfloat16):
    """Load ZImage pipeline."""
    print(f"Loading model from {model_path}...")
    pipe = ZImagePipeline.from_pretrained(model_path, torch_dtype=dtype, low_cpu_mem_usage=False)
    pipe.to(device)
    print("Model loaded.")
    return pipe


def register_qwen_classes():
    import transformers.utils as _tu

    if not hasattr(_tu, "FLAX_WEIGHTS_NAME"):
        _tu.FLAX_WEIGHTS_NAME = "flax_model.msgpack"

    import diffusers

    if hasattr(diffusers, "QwenImagePipeline"):
        return

    from models.qwen_ip.pipeline_qwenimage import QwenImagePipeline
    from models.qwen_ip.transformer import QwenTransformer2DModel
    from models.qwen_ip.autoencoder_kl_qwenimage import AutoencoderKLQwenImage

    diffusers.QwenImagePipeline = QwenImagePipeline
    diffusers.QwenImageTransformer2DModel = QwenTransformer2DModel
    diffusers.AutoencoderKLQwenImage = AutoencoderKLQwenImage


def load_qwen_pipeline(model_path: str, device: str, dtype=torch.bfloat16):
    """Load QwenImage pipeline."""
    import diffusers

    register_qwen_classes()
    print(f"Loading QwenImage model from {model_path}...")
    pipe = diffusers.DiffusionPipeline.from_pretrained(model_path, torch_dtype=dtype, low_cpu_mem_usage=False)
    pipe.to("cpu")
    if isinstance(device, str) and device.startswith("cuda"):
        gpu_id = int(device.split(":", 1)[1]) if ":" in device else 0
        pipe.enable_model_cpu_offload(gpu_id=gpu_id)
    print("QwenImage model loaded.")
    return pipe


def load_pipeline(backend: str, model_path: str, device: str, dtype=torch.bfloat16):
    if backend == "qwen":
        return load_qwen_pipeline(model_path, device, dtype=dtype)
    return load_zimage_pipeline(model_path, device, dtype=dtype)


def cuda_empty_cache(device: str):
    if isinstance(device, str) and device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()


def offload_qwen_pipeline(pipeline):
    if hasattr(pipeline, "maybe_free_model_hooks"):
        pipeline.maybe_free_model_hooks()
    cuda_empty_cache(getattr(pipeline, "_execution_device", "cuda"))


def prepare_noise(pipeline, height: int, width: int, device: str, generator=None):
    """Prepare shared noise"""
    latent_h = 2 * (height // (pipeline.vae_scale_factor * 2))
    latent_w = 2 * (width // (pipeline.vae_scale_factor * 2))
    channels = pipeline.transformer.in_channels
    return torch.randn((1, channels, latent_h, latent_w), generator=generator, device=device, dtype=torch.float32)


def prepare_qwen_noise(pipeline, height: int, width: int, device: str, generator=None):
    """Prepare shared noise for Qwen packed latents."""
    latent_h = 2 * (height // (pipeline.vae_scale_factor * 2))
    latent_w = 2 * (width // (pipeline.vae_scale_factor * 2))
    channels = pipeline.transformer.config.in_channels // 4
    return torch.randn((1, 1, channels, latent_h, latent_w), generator=generator, device=device, dtype=torch.float32)


def get_qwen_timesteps(pipeline, noise: torch.Tensor, steps: int, device: str):
    latent_h = noise.shape[3]
    latent_w = noise.shape[4]
    num_channels = noise.shape[2]
    packed_noise = pipeline._pack_latents(noise, 1, num_channels, latent_h, latent_w)
    sigmas = np.linspace(1.0, 1 / steps, steps)
    image_seq_len = packed_noise.shape[1]
    mu = calculate_qwen_shift(
        image_seq_len,
        pipeline.scheduler.config.get("base_image_seq_len", 256),
        pipeline.scheduler.config.get("max_image_seq_len", 4096),
        pipeline.scheduler.config.get("base_shift", 0.5),
        pipeline.scheduler.config.get("max_shift", 1.15),
    )
    timesteps, _ = retrieve_qwen_timesteps(
        pipeline.scheduler,
        steps,
        device,
        sigmas=sigmas,
        mu=mu,
    )
    return timesteps, packed_noise


def decode_latent(pipeline, latent: torch.Tensor) -> Image.Image:
    """Decode latent to PIL Image"""
    latent = latent.to(pipeline.vae.dtype)
    latent = (latent / pipeline.vae.config.scaling_factor) + pipeline.vae.config.shift_factor
    with torch.no_grad():
        image = pipeline.vae.decode(latent, return_dict=False)[0]
    return pipeline.image_processor.postprocess(image, output_type="pil")[0]


def decode_latent_qwen(pipeline, latent: torch.Tensor, height: int, width: int) -> Image.Image:
    """Decode packed Qwen latent to PIL Image."""
    latent = pipeline._unpack_latents(latent, height, width, pipeline.vae_scale_factor)
    latent = latent.to(pipeline.vae.dtype)
    latents_mean = torch.tensor(pipeline.vae.config.latents_mean).view(
        1, pipeline.vae.config.z_dim, 1, 1, 1
    ).to(latent.device, latent.dtype)
    latents_std = 1.0 / torch.tensor(pipeline.vae.config.latents_std).view(
        1, pipeline.vae.config.z_dim, 1, 1, 1
    ).to(latent.device, latent.dtype)
    latent = latent / latents_std + latents_mean
    with torch.no_grad():
        image = pipeline.vae.decode(latent, return_dict=False)[0][:, :, 0]
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


def run_pass1_reference_qwen(
    pipeline,
    prompt: str,
    packed_noise: torch.Tensor,
    args,
    generator,
) -> Image.Image:
    """Pass 1: Qwen direct pipeline generation."""
    return pipeline(
        prompt=prompt,
        negative_prompt="",
        true_cfg_scale=args.qwen_true_cfg_scale,
        guidance_scale=args.qwen_guidance_scale,
        height=args.height,
        width=args.width,
        num_inference_steps=args.steps,
        generator=generator,
        latents=packed_noise.clone(),
    ).images[0]


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


@torch.no_grad()
def run_pass2_injection_qwen(
    pipeline,
    prompt: str,
    noise: torch.Tensor,
    timesteps,
    injection_data: dict,
    device: str,
    args,
) -> torch.Tensor:
    """Pass 2: Qwen denoising with glyph injection."""
    execution_device = pipeline._execution_device
    dtype = pipeline.transformer.dtype
    latent_h = noise.shape[3]
    latent_w = noise.shape[4]
    num_channels = noise.shape[2]

    prompt_embeds, _ = pipeline.encode_prompt(prompt=prompt, device=execution_device, max_sequence_length=512)
    negative_embeds, _ = pipeline.encode_prompt(prompt="", device=execution_device, max_sequence_length=512)
    latent = pipeline._pack_latents(
        noise.to(device=execution_device, dtype=dtype),
        1,
        num_channels,
        latent_h,
        latent_w,
    )
    img_shapes = [[(1, latent_h // 2, latent_w // 2)]]
    txt_seq_lens = [prompt_embeds.shape[1]]
    neg_txt_seq_lens = [negative_embeds.shape[1]]

    sigmas = np.linspace(1.0, 1 / args.steps, args.steps)
    image_seq_len = latent.shape[1]
    mu = calculate_qwen_shift(
        image_seq_len,
        pipeline.scheduler.config.get("base_image_seq_len", 256),
        pipeline.scheduler.config.get("max_image_seq_len", 4096),
        pipeline.scheduler.config.get("base_shift", 0.5),
        pipeline.scheduler.config.get("max_shift", 1.15),
    )
    timesteps, _ = retrieve_qwen_timesteps(
        pipeline.scheduler,
        args.steps,
        execution_device,
        sigmas=sigmas,
        mu=mu,
    )

    guidance = None
    if pipeline.transformer.config.guidance_embeds and args.qwen_guidance_scale is not None:
        guidance = torch.full([1], args.qwen_guidance_scale, device=execution_device, dtype=torch.float32)

    pipeline.scheduler.set_begin_index(0)
    for step_idx, t in enumerate(timesteps):
        timestep = t.expand(latent.shape[0]).to(latent.dtype)
        noise_pred = pipeline.transformer(
            hidden_states=latent.to(dtype),
            timestep=timestep / 1000,
            guidance=guidance,
            encoder_hidden_states=prompt_embeds,
            txt_seq_lens=txt_seq_lens,
            img_shapes=img_shapes,
            return_dict=False,
        )[0]

        if args.qwen_true_cfg_scale > 1:
            neg_pred = pipeline.transformer(
                hidden_states=latent.to(dtype),
                timestep=timestep / 1000,
                guidance=guidance,
                encoder_hidden_states=negative_embeds,
                txt_seq_lens=neg_txt_seq_lens,
                img_shapes=img_shapes,
                return_dict=False,
            )[0]
            combined = neg_pred + args.qwen_true_cfg_scale * (noise_pred - neg_pred)
            cond_norm = torch.norm(noise_pred, dim=-1, keepdim=True)
            comb_norm = torch.norm(combined, dim=-1, keepdim=True).clamp_min(1e-6)
            noise_pred = combined * (cond_norm / comb_norm)

        latent = pipeline.scheduler.step(noise_pred, t, latent, return_dict=False)[0]

        if "glyph_injector" in injection_data:
            spatial = pipeline._unpack_latents(latent, args.height, args.width, pipeline.vae_scale_factor)
            spatial = injection_data["glyph_injector"].inject_latent(spatial.squeeze(2), injection_data, step_idx + 1)
            latent = pipeline._pack_latents(spatial.unsqueeze(2), 1, num_channels, latent_h, latent_w)

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
    vlm_agent=None, output_path: str = None, enable_cpu_offload: bool = False,
) -> Image.Image:
    """Pass 3: FluxKlein stylization - multi-variant generation + VLM selection"""
    from models.fluxklein.inference_fluxklein import FluxKleinGenerator

    print(f"Loading FluxKlein from {klein_model_path}...")
    klein = FluxKleinGenerator(
        model_path=klein_model_path,
        device=device,
        enable_cpu_offload=enable_cpu_offload,
    )

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

    device = args.device
    model_path = resolve_model_path(args.backend, args.model_path)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    pipeline = load_pipeline(args.backend, model_path, device)

    print("Initializing VLM Agent...")
    vlm_agent = VLMAgent()
    print("Initializing Glyph Injector...")
    glyph_injector = create_glyph_injector(pipeline, device=device)

    working_prompt = args.prompt + ",horizontal text layout."
    print(f"Working prompt: {working_prompt}")

    if args.backend == "qwen":
        noise = prepare_qwen_noise(pipeline, args.height, args.width, device, generator)
        timesteps, packed_noise = get_qwen_timesteps(pipeline, noise, args.steps, device)
    else:
        noise = prepare_noise(pipeline, args.height, args.width, device, generator)
        pipeline.scheduler.set_timesteps(args.steps, device=device)
        timesteps = pipeline.scheduler.timesteps
        packed_noise = None

    print("\n=== Pass 1: Reference Image ===")
    if args.backend == "qwen":
        reference_image = run_pass1_reference_qwen(pipeline, working_prompt, packed_noise, args, generator)
        offload_qwen_pipeline(pipeline)
    else:
        reference_image = run_pass1_reference(pipeline, working_prompt, noise.clone(), timesteps, device)

    print("\n=== VLM Typography Planning ===")
    typography_plan = vlm_agent.analyze_typography(reference_image, working_prompt, args.text)

    print("\n=== Pass 2: Glyph Injection ===")
    if args.backend == "zimage":
        pipeline.scheduler.set_timesteps(args.steps, device=device)
        timesteps = pipeline.scheduler.timesteps

    clean_prompt = vlm_agent.generate_clean_prompt(working_prompt, typography_plan)
    print(f"Clean prompt: {clean_prompt}")

    image_size = (args.width, args.height)
    injection_noise = noise.squeeze(1) if args.backend == "qwen" else noise
    injection_data = glyph_injector.prepare_injection_from_plan(
        typography_plan,
        image_size,
        injection_noise,
        timesteps,
    )
    if args.backend == "qwen":
        offload_qwen_pipeline(pipeline)

    if injection_data["full_mask"].max() == 0:
        print("Warning: Empty glyph mask, returning reference image")
        reference_image.save(args.output)
        return

    injection_data["glyph_injector"] = glyph_injector
    if args.backend == "qwen":
        background_latent = run_pass2_injection_qwen(pipeline, clean_prompt, noise, timesteps, injection_data, device, args)
        background = decode_latent_qwen(pipeline, background_latent, args.height, args.width)
    else:
        background_latent = run_pass2_injection(pipeline, clean_prompt, noise, timesteps, injection_data, device)
        background = decode_latent(pipeline, background_latent)

    pass2_image = pixel_composite_text(background, injection_data)
    print("Pass 2 completed.")

    if not args.no_harmonize:
        print("\n=== Pass 3: Harmonization ===")
        if args.backend == "qwen":
            offload_qwen_pipeline(pipeline)
        else:
            pipeline.to("cpu")
            cuda_empty_cache(device)

        final_image = run_pass3_klein(
            pass2_image, injection_data, args.klein_model_path,
            working_prompt, args.klein_steps, args.klein_guidance, args.seed, device,
            vlm_agent=vlm_agent,
            output_path=args.output,
            enable_cpu_offload=(args.backend == "qwen"),
        )

        if args.backend != "qwen":
            pipeline.to(device)
    else:
        final_image = pass2_image

    if final_image.mode != "RGB":
        final_image = final_image.convert("RGB")
    final_image.save(args.output)
    print(f"\nSaved to: {args.output}")


if __name__ == "__main__":
    main()
