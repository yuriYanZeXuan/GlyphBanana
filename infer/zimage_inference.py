"""ZImage three-pass inference backend."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

os.environ["TORCH_COMPILE_DISABLE"] = "1"

import numpy as np
import torch
from PIL import Image

from infer.VLM_agent import VLMAgent
from infer.generation_utils import DEFAULT_KLEIN_MODEL_PATH, run_pass3_klein, text_regions_to_plan
from infer.glyph_injector import create_glyph_injector
from train.zimage_ip.pipeline_z_image import ZImagePipeline

DEFAULT_MODEL_PATH = "/mnt/tidalfs-bdsz01/usr/tusen/yanzexuan/weight/Z-Image"


@dataclass
class ZImageGenerationConfig:
    height: int = 1024
    width: int = 1024
    num_inference_steps: int = 20
    seed: Optional[int] = 42
    use_prompt_refiner: bool = True
    use_harmonization: bool = True
    klein_model_path: str = DEFAULT_KLEIN_MODEL_PATH
    klein_steps: int = 10
    klein_guidance_scale: float = 4.0


def load_pipeline(model_path: str, device: str, dtype=torch.bfloat16):
    """Load ZImage pipeline."""
    print(f"Loading model from {model_path}...")
    pipe = ZImagePipeline.from_pretrained(model_path, torch_dtype=dtype, low_cpu_mem_usage=False)
    pipe.to(device)
    print("Model loaded.")
    return pipe


def prepare_noise(pipeline, height: int, width: int, device: str, generator=None):
    """Prepare shared noise."""
    latent_h = 2 * (height // (pipeline.vae_scale_factor * 2))
    latent_w = 2 * (width // (pipeline.vae_scale_factor * 2))
    channels = pipeline.transformer.in_channels
    return torch.randn((1, channels, latent_h, latent_w), generator=generator, device=device, dtype=torch.float32)


def decode_latent(pipeline, latent: torch.Tensor) -> Image.Image:
    """Decode latent to PIL image."""
    latent = latent.to(pipeline.vae.dtype)
    latent = (latent / pipeline.vae.config.scaling_factor) + pipeline.vae.config.shift_factor
    with torch.no_grad():
        image = pipeline.vae.decode(latent, return_dict=False)[0]
    return pipeline.image_processor.postprocess(image, output_type="pil")[0]


def run_pass1_reference(pipeline, prompt: str, noise: torch.Tensor, timesteps, device: str) -> Image.Image:
    """Pass 1 reference image generation."""
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


def run_pass2_injection(pipeline, prompt: str, noise: torch.Tensor, timesteps, injection_data: dict, device: str):
    """Pass 2 denoising with glyph injection."""
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

        injector = injection_data.get("glyph_injector")
        if injector is not None:
            latent = injector.inject_latent(latent, injection_data, step_idx + 1)

    return latent


def pixel_composite_text(background: Image.Image, injection_data: dict) -> Image.Image:
    """Composite rendered text in pixel space."""
    template = injection_data["combined_template"]
    mask = injection_data["full_mask"]

    bg_arr = np.array(background)
    tpl_arr = np.array(template)

    if tpl_arr.shape[:2] != bg_arr.shape[:2]:
        template = template.resize(background.size, Image.LANCZOS)
        tpl_arr = np.array(template)
    if mask.shape[:2] != bg_arr.shape[:2]:
        mask = np.array(Image.fromarray(mask).resize(background.size, Image.NEAREST))

    result = bg_arr.copy()
    result[mask > 127] = tpl_arr[mask > 127]
    return Image.fromarray(result)


class ZImageInference:
    """Reusable ZImage three-pass generator."""

    def __init__(self, model_path: str = DEFAULT_MODEL_PATH, device: str = "cuda", dtype=torch.bfloat16):
        self.model_path = model_path
        self.device = device
        self.dtype = dtype
        self.pipeline = load_pipeline(model_path, device, dtype=dtype)
        print("Initializing VLM Agent...")
        self.vlm_agent = VLMAgent()
        print("Initializing Glyph Injector...")
        self.glyph_injector = create_glyph_injector(self.pipeline, device=device)

    def generate(
        self,
        prompt: str,
        text_contents: list[str],
        text_regions: Optional[list[dict]] = None,
        config: Optional[ZImageGenerationConfig] = None,
        output_path: Optional[str] = None,
    ) -> Image.Image:
        config = config or ZImageGenerationConfig()
        working_prompt = prompt + ",horizontal text layout." if config.use_prompt_refiner else prompt
        generator = None
        if config.seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(config.seed)
        reference_image = None

        if text_regions:
            typography_plan = text_regions_to_plan(text_regions)
        else:
            noise_for_plan = prepare_noise(self.pipeline, config.height, config.width, self.device, generator)
            self.pipeline.scheduler.set_timesteps(config.num_inference_steps, device=self.device)
            reference_image = run_pass1_reference(
                self.pipeline,
                working_prompt,
                noise_for_plan.clone(),
                self.pipeline.scheduler.timesteps,
                self.device,
            )
            typography_plan = self.vlm_agent.analyze_typography(reference_image, working_prompt, text_contents)

        noise = prepare_noise(self.pipeline, config.height, config.width, self.device, generator)
        self.pipeline.scheduler.set_timesteps(config.num_inference_steps, device=self.device)
        timesteps = self.pipeline.scheduler.timesteps

        clean_prompt = self.vlm_agent.generate_clean_prompt(working_prompt, typography_plan)
        print(f"Clean prompt: {clean_prompt}")

        injection_data = self.glyph_injector.prepare_injection_from_plan(
            typography_plan,
            (config.width, config.height),
            noise,
            timesteps,
        )

        if injection_data["full_mask"].max() == 0:
            print("Warning: Empty glyph mask, falling back to prompt-only result.")
            if reference_image is not None:
                return reference_image
            self.pipeline.scheduler.set_timesteps(config.num_inference_steps, device=self.device)
            return run_pass1_reference(
                self.pipeline,
                working_prompt,
                noise.clone(),
                self.pipeline.scheduler.timesteps,
                self.device,
            )

        injection_data["glyph_injector"] = self.glyph_injector
        background_latent = run_pass2_injection(
            self.pipeline,
            clean_prompt,
            noise,
            timesteps,
            injection_data,
            self.device,
        )
        background = decode_latent(self.pipeline, background_latent)
        pass2_image = pixel_composite_text(background, injection_data)

        if not config.use_harmonization:
            return pass2_image.convert("RGB") if pass2_image.mode != "RGB" else pass2_image

        self.pipeline.to("cpu")
        torch.cuda.empty_cache()
        final_image = run_pass3_klein(
            pass2_image,
            injection_data,
            config.klein_model_path,
            working_prompt,
            config.klein_steps,
            config.klein_guidance_scale,
            config.seed or 42,
            self.device,
            vlm_agent=self.vlm_agent,
            output_path=output_path,
        )
        self.pipeline.to(self.device)
        return final_image.convert("RGB") if final_image.mode != "RGB" else final_image


ZImageGenerator = ZImageInference
