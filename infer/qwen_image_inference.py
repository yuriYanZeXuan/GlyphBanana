"""
Qwen-Image 三阶段推理后端。

对齐 GlyphBanana 的 zimage 入口，支持：
1. Pass 1 参考图 + VLM 排版规划
2. Pass 2 clean prompt + glyph injection
3. Pass 3 FluxKlein harmonization
4. 直接传入 text_regions 跳过 layout planner
"""

import inspect
import os
from dataclasses import dataclass
from typing import Optional

os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = "1"

import numpy as np
import torch
from PIL import Image

from infer.generation_utils import DEFAULT_KLEIN_MODEL_PATH, run_pass3_klein, text_regions_to_plan
from infer.VLM_agent import VLMAgent
from infer.glyph_injector import GlyphInjector

DEFAULT_MODEL_PATH = "/mnt/tidalfs-bdsz01/usr/tusen/yanzexuan/weight/qwen-image-2512"


def _calculate_shift(
    image_seq_len,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    return image_seq_len * m + b


def _retrieve_timesteps(scheduler, num_inference_steps=None, device=None, timesteps=None, sigmas=None, **kwargs):
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed.")
    if timesteps is not None:
        if "timesteps" not in inspect.signature(scheduler.set_timesteps).parameters:
            raise ValueError(f"{scheduler.__class__} does not support custom timesteps.")
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        return scheduler.timesteps, len(scheduler.timesteps)
    if sigmas is not None:
        if "sigmas" not in inspect.signature(scheduler.set_timesteps).parameters:
            raise ValueError(f"{scheduler.__class__} does not support custom sigmas.")
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        return scheduler.timesteps, len(scheduler.timesteps)
    scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
    return scheduler.timesteps, num_inference_steps


@dataclass
class QwenGenerationConfig:
    height: int = 1024
    width: int = 1024
    num_inference_steps: int = 28
    true_cfg_scale: float = 4.0
    guidance_scale: Optional[float] = None
    seed: Optional[int] = 42
    use_prompt_refiner: bool = True
    use_harmonization: bool = True
    klein_model_path: str = DEFAULT_KLEIN_MODEL_PATH
    klein_steps: int = 10
    klein_guidance_scale: float = 4.0


class QwenImageInference:
    """Qwen-Image 推理后端。"""

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_PATH,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.model_path = model_path
        self.device = device
        self.dtype = dtype
        self._pipeline = None
        self._vlm_agent = None
        self._glyph_injector = None

    @staticmethod
    def _register_qwen_classes():
        import transformers.utils as _tu

        if not hasattr(_tu, "FLAX_WEIGHTS_NAME"):
            _tu.FLAX_WEIGHTS_NAME = "flax_model.msgpack"

        import diffusers

        if hasattr(diffusers, "QwenImagePipeline"):
            return

        from train.qwen_ip.autoencoder_kl_qwenimage import AutoencoderKLQwenImage
        from train.qwen_ip.pipeline_qwenimage import QwenImagePipeline
        from train.qwen_ip.transformer import QwenTransformer2DModel

        diffusers.QwenImagePipeline = QwenImagePipeline
        diffusers.QwenImageTransformer2DModel = QwenTransformer2DModel
        diffusers.AutoencoderKLQwenImage = AutoencoderKLQwenImage

    @property
    def pipeline(self):
        if self._pipeline is None:
            self._register_qwen_classes()
            import diffusers

            print(f"Loading Qwen-Image from {self.model_path}...")
            pipe = diffusers.DiffusionPipeline.from_pretrained(self.model_path, torch_dtype=self.dtype)
            pipe.enable_model_cpu_offload(
                gpu_id=int(self.device.split(":")[-1]) if ":" in self.device else 0
            )
            self._pipeline = pipe
            print("Qwen-Image loaded.")
        return self._pipeline

    @property
    def vlm_agent(self) -> VLMAgent:
        if self._vlm_agent is None:
            self._vlm_agent = VLMAgent()
        return self._vlm_agent

    @property
    def glyph_injector(self) -> GlyphInjector:
        if self._glyph_injector is None:
            self._glyph_injector = GlyphInjector(
                vae=self.pipeline.vae,
                scheduler=self.pipeline.scheduler,
                device=self.device,
                dtype=self.dtype,
            )
        return self._glyph_injector

    def generate(
        self,
        prompt: str,
        text_contents: Optional[list[str]] = None,
        text_regions: Optional[list[dict]] = None,
        config: Optional[QwenGenerationConfig] = None,
        output_path: Optional[str] = None,
    ) -> Image.Image:
        config = config or QwenGenerationConfig()
        working_prompt = prompt + ",horizontal text layout." if config.use_prompt_refiner else prompt
        generator = None
        if config.seed is not None:
            generator = torch.Generator(device="cpu").manual_seed(config.seed)

        reference_image = None
        if text_regions:
            typography_plan = text_regions_to_plan(text_regions)
        else:
            print("=== Pass 1: Reference Image ===")
            reference_image = self.pipeline(
                prompt=working_prompt,
                height=config.height,
                width=config.width,
                num_inference_steps=config.num_inference_steps,
                true_cfg_scale=config.true_cfg_scale,
                guidance_scale=config.guidance_scale,
                generator=generator,
            ).images[0]
            print("=== VLM Typography Planning ===")
            typography_plan = self.vlm_agent.analyze_typography(reference_image, working_prompt, text_contents or [])

        print("=== Pass 2: Glyph Injection ===")
        clean_prompt = self.vlm_agent.generate_clean_prompt(working_prompt, typography_plan)
        print(f"Clean prompt: {clean_prompt}")

        pipe = self.pipeline
        latent_h = 2 * (config.height // (pipe.vae_scale_factor * 2))
        latent_w = 2 * (config.width // (pipe.vae_scale_factor * 2))
        num_channels = pipe.transformer.config.in_channels // 4
        noise = torch.randn((1, 1, num_channels, latent_h, latent_w), generator=generator, device="cpu", dtype=torch.float32)

        packed_noise = pipe._pack_latents(noise, 1, num_channels, latent_h, latent_w)
        sigmas = np.linspace(1.0, 1 / config.num_inference_steps, config.num_inference_steps)
        mu = _calculate_shift(
            packed_noise.shape[1],
            pipe.scheduler.config.get("base_image_seq_len", 256),
            pipe.scheduler.config.get("max_image_seq_len", 4096),
            pipe.scheduler.config.get("base_shift", 0.5),
            pipe.scheduler.config.get("max_shift", 1.15),
        )
        timesteps, _ = _retrieve_timesteps(pipe.scheduler, config.num_inference_steps, "cpu", sigmas=sigmas, mu=mu)

        injection_data = self.glyph_injector.prepare_injection_from_plan(
            typography_plan,
            (config.width, config.height),
            noise.squeeze(1),
            timesteps,
        )

        if injection_data["full_mask"].max() == 0:
            print("Warning: Empty glyph mask, falling back to prompt-only result.")
            if reference_image is not None:
                return reference_image
            return self.pipeline(
                prompt=working_prompt,
                height=config.height,
                width=config.width,
                num_inference_steps=config.num_inference_steps,
                true_cfg_scale=config.true_cfg_scale,
                guidance_scale=config.guidance_scale,
                generator=generator,
            ).images[0]

        background = self._run_pass2_denoising(clean_prompt, noise, config, injection_data)
        pass2_image = self._pixel_composite_text(background, injection_data)

        if not config.use_harmonization:
            return pass2_image.convert("RGB") if pass2_image.mode != "RGB" else pass2_image

        print("=== Pass 3: Harmonization ===")
        self._offload_pipeline()
        final_image = self._run_pass3_klein(pass2_image, injection_data, working_prompt, config, output_path)
        return final_image.convert("RGB") if final_image.mode != "RGB" else final_image

    @torch.no_grad()
    def _run_pass2_denoising(self, clean_prompt, noise, config, injection_data):
        pipe = self.pipeline
        device = pipe._execution_device
        dtype = pipe.transformer.dtype
        latent_h = noise.shape[3]
        latent_w = noise.shape[4]
        num_channels = noise.shape[2]

        prompt_embeds, _ = pipe.encode_prompt(clean_prompt, device=device, max_sequence_length=512)
        negative_embeds, _ = pipe.encode_prompt("", device=device, max_sequence_length=512)

        if getattr(pipe, "text_encoder", None) is not None:
            pipe.text_encoder.to("cpu")
        torch.cuda.empty_cache()

        latent = pipe._pack_latents(noise.to(device=device, dtype=dtype), 1, num_channels, latent_h, latent_w)
        sigmas = np.linspace(1.0, 1 / config.num_inference_steps, config.num_inference_steps)
        mu = _calculate_shift(
            latent.shape[1],
            pipe.scheduler.config.get("base_image_seq_len", 256),
            pipe.scheduler.config.get("max_image_seq_len", 4096),
            pipe.scheduler.config.get("base_shift", 0.5),
            pipe.scheduler.config.get("max_shift", 1.15),
        )
        timesteps, _ = _retrieve_timesteps(pipe.scheduler, config.num_inference_steps, device, sigmas=sigmas, mu=mu)
        img_shapes = [[(1, latent_h // 2, latent_w // 2)]]
        txt_seq_lens = [prompt_embeds.shape[1]]
        neg_txt_seq_lens = [negative_embeds.shape[1]]

        guidance = None
        if pipe.transformer.config.guidance_embeds and config.guidance_scale is not None:
            guidance = torch.full([1], config.guidance_scale, device=device, dtype=torch.float32)

        pipe.scheduler.set_begin_index(0)
        for step_idx, timestep in enumerate(timesteps):
            expanded = timestep.expand(latent.shape[0]).to(latent.dtype)

            noise_pred = pipe.transformer(
                hidden_states=latent.to(dtype),
                timestep=expanded / 1000,
                guidance=guidance,
                encoder_hidden_states=prompt_embeds,
                txt_seq_lens=txt_seq_lens,
                img_shapes=img_shapes,
                return_dict=False,
            )[0]

            if config.true_cfg_scale > 1:
                neg_pred = pipe.transformer(
                    hidden_states=latent.to(dtype),
                    timestep=expanded / 1000,
                    guidance=guidance,
                    encoder_hidden_states=negative_embeds,
                    txt_seq_lens=neg_txt_seq_lens,
                    img_shapes=img_shapes,
                    return_dict=False,
                )[0]
                combined = neg_pred + config.true_cfg_scale * (noise_pred - neg_pred)
                cond_norm = torch.norm(noise_pred, dim=-1, keepdim=True)
                comb_norm = torch.norm(combined, dim=-1, keepdim=True).clamp(min=1e-6)
                noise_pred = combined * (cond_norm / comb_norm)

            latent = pipe.scheduler.step(noise_pred, timestep, latent, return_dict=False)[0]

            spatial = pipe._unpack_latents(latent, config.height, config.width, pipe.vae_scale_factor)
            spatial_4d = spatial.squeeze(2)
            spatial_4d = self.glyph_injector.inject_latent(spatial_4d, injection_data, step_idx + 1)
            latent = pipe._pack_latents(spatial_4d.unsqueeze(2), 1, num_channels, latent_h, latent_w)

        pipe.transformer.to("cpu")
        torch.cuda.empty_cache()
        return self._decode_latent(latent, config.height, config.width)

    def _decode_latent(self, latent, height, width):
        pipe = self.pipeline
        device = latent.device
        latent = pipe._unpack_latents(latent, height, width, pipe.vae_scale_factor)
        pipe.vae.to(device)
        latent = latent.to(pipe.vae.dtype)

        latents_mean = torch.tensor(pipe.vae.config.latents_mean).view(1, pipe.vae.config.z_dim, 1, 1, 1).to(device, latent.dtype)
        latents_std_inv = 1.0 / torch.tensor(pipe.vae.config.latents_std).view(1, pipe.vae.config.z_dim, 1, 1, 1).to(device, latent.dtype)
        latent = latent / latents_std_inv + latents_mean

        with torch.no_grad():
            image = pipe.vae.decode(latent, return_dict=False)[0][:, :, 0]
        return pipe.image_processor.postprocess(image, output_type="pil")[0]

    def _pixel_composite_text(self, background, injection_data):
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

    def _offload_pipeline(self):
        if self._pipeline is None:
            return
        for name in ("transformer", "vae", "text_encoder"):
            component = getattr(self._pipeline, name, None)
            if component is not None:
                component.to("cpu")
        self._glyph_injector = None
        torch.cuda.empty_cache()

    def _run_pass3_klein(self, pass2_image, injection_data, prompt, config, output_path=None):
        return run_pass3_klein(
            pass2_image=pass2_image,
            injection_data=injection_data,
            klein_model_path=config.klein_model_path,
            prompt=prompt,
            steps=config.klein_steps,
            guidance=config.klein_guidance_scale,
            seed=config.seed or 42,
            device=self.device,
            vlm_agent=self.vlm_agent,
            output_path=output_path,
        )
