"""
Glyph Injector: Text Rendering and Latent Injection Interface

Implementation Flow:
1. Render text with system font on white background with black text
2. Extract text mask using Otsu's method binarization
3. Perform flow matching inversion on text template, save latent list
4. Inject latent at corresponding timestep during denoising
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from typing import Optional, Tuple

from .formula_helper import render_formula, resolve_font_name


class GlyphInjector:
    """Text Injector: Implements text template rendering, mask extraction, latent inversion and latent injection"""
    
    def __init__(
        self, 
        vae,
        scheduler,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.vae = vae
        self.scheduler = scheduler
        self.device = device
        self.dtype = dtype
        
        # VAE scale factor
        if hasattr(vae, 'config') and hasattr(vae.config, 'block_out_channels'):
            self.vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
        else:
            self.vae_scale_factor = getattr(vae, 'spatial_compression_ratio', 8)
    
    # Color name -> hex mapping
    _COLOR_MAP = {
        "white": "#FFFFFF", "black": "#FFFFFF", "red": "#FF4444",
        "blue": "#6688FF", "green": "#44DD44", "yellow": "#FFEE44",
        "orange": "#FFAA33", "brown": "#CC9966", "gray": "#BBBBBB",
        "gold": "#FFD700", "silver": "#C0C0C0", "purple": "#BB77FF",
        "pink": "#FF88BB",
    }

    @classmethod
    def _resolve_color(cls, color: str) -> str:
        """Map color name/hex to hex value usable for template (visible on black background)"""
        c = color.strip().lower()
        if c in cls._COLOR_MAP:
            return cls._COLOR_MAP[c]
        if c.startswith("#"):
            hex_val = c.lstrip("#")
            if len(hex_val) >= 6:
                r, g, b = int(hex_val[0:2], 16), int(hex_val[2:4], 16), int(hex_val[4:6], 16)
                if r + g + b < 128:
                    return "#FFFFFF"
            return color
        return "#FFFFFF"

    def render_text_template(
        self,
        text: str,
        width: int,
        height: int,
        text_color: str = "white",
        force_latex: bool = False,
        font_weight: str = "regular",
        font_path: Optional[str] = None,
        rotation: float = 0.0,
    ) -> Image.Image:
        """Render text template image (black background + specified text color)"""
        return render_formula(
            text, width, height, text_color,
            force_latex, font_weight=font_weight, font_path=font_path,
            rotation=rotation,
        )
    
    def extract_text_mask(self, image: np.ndarray) -> np.ndarray:
        """Extract text mask using Otsu's method, text regions are 255"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        mean_val = np.mean(gray)
        
        if mean_val < 127:
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            white_ratio = np.sum(binary == 255) / binary.size
            text_mask = binary if white_ratio < 0.5 else cv2.bitwise_not(binary)
        else:
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            black_ratio = np.sum(binary == 255) / binary.size
            text_mask = binary if black_ratio < 0.5 else cv2.bitwise_not(binary)
        
        kernel = np.ones((2, 2), np.uint8)
        text_mask = cv2.morphologyEx(text_mask, cv2.MORPH_CLOSE, kernel)
        
        return text_mask
    
    @property
    def _is_3d_vae(self) -> bool:
        return hasattr(self.vae, 'spatial_compression_ratio')

    def encode_image(self, image: Image.Image) -> torch.Tensor:
        """Encode PIL Image to latent"""
        img_array = np.array(image).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0)
        img_tensor = img_tensor * 2.0 - 1.0
        img_tensor = img_tensor.to(device=self.device, dtype=self.dtype)

        if self._is_3d_vae:
            img_tensor = img_tensor.unsqueeze(2)

        with torch.no_grad():
            latent = self.vae.encode(img_tensor).latent_dist.sample()
            if hasattr(self.vae.config, 'shift_factor'):
                latent = (latent - self.vae.config.shift_factor) * self.vae.config.scaling_factor

        if self._is_3d_vae:
            latent = latent.squeeze(2)

        return latent
    
    def decode_latent(self, latent: torch.Tensor) -> Image.Image:
        """Decode latent to PIL Image"""
        with torch.no_grad():
            latent = latent.to(self.vae.dtype)
            if hasattr(self.vae.config, 'shift_factor'):
                latent = (latent / self.vae.config.scaling_factor) + self.vae.config.shift_factor

            if self._is_3d_vae:
                latent = latent.unsqueeze(2)

            image = self.vae.decode(latent, return_dict=False)[0]

            if self._is_3d_vae:
                image = image[:, :, 0]

        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().float().permute(0, 2, 3, 1).numpy()[0]
        return Image.fromarray((image * 255).astype(np.uint8))
    
    @staticmethod
    def _freq_decompose_inject(
        current: torch.Tensor,
        template: torch.Tensor,
        mask: torch.Tensor,
        strength: float,
        kernel_size: int = 5,
    ):
        """Frequency decomposition injection: only inject high-frequency components from template, preserve low-frequency components from current"""
        pad = kernel_size // 2

        def blur(x):
            return F.avg_pool2d(
                F.pad(x, [pad] * 4, mode="reflect"),
                kernel_size, stride=1,
            )

        template_lf = blur(template)
        template_hf = template - template_lf
        current_lf = blur(current)
        current_hf = current - current_lf

        blended_hf = current_hf * (1 - mask * strength) + template_hf * (mask * strength)
        return current_lf + blended_hf
    
    def compute_inversion_latents(
        self, 
        latent_0: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor
    ) -> list[torch.Tensor]:
        """Compute latent list for flow matching inversion
        
        Flow matching: z_t = (1 - sigma) * z_0 + sigma * noise
        """
        noise = noise.to(device=latent_0.device, dtype=latent_0.dtype)
        latent_list = []
        for t in timesteps:
            sigma = t.float() / 1000.0
            z_t = (1 - sigma) * latent_0 + sigma * noise
            latent_list.append(z_t.clone())
        
        latent_list.append(latent_0.clone())
        return latent_list
    
    def prepare_injection_from_plan(
        self,
        typography_plan: dict,
        image_size: Tuple[int, int],
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> dict:
        """Render glyph templates according to typography plan and prepare injection data"""
        width, height = image_size
        combined_template = Image.new("RGB", (width, height), "black")
        
        for region_spec in typography_plan.get("text_regions", []):
            content = region_spec["content"]
            bbox = region_spec["bbox"]
            color = self._resolve_color(region_spec.get("color", "white"))

            x1 = int(bbox[0] * width)
            y1 = int(bbox[1] * height)
            x2 = int(bbox[2] * width)
            y2 = int(bbox[3] * height)
            region_width = max(x2 - x1, 1)
            region_height = max(y2 - y1, 1)

            font_path = region_spec.get("font_path") or resolve_font_name(region_spec.get("font"))

            text_img = self.render_text_template(
                content, region_width, region_height,
                text_color=color,
                force_latex=region_spec.get("is_latex", False),
                font_weight=region_spec.get("font_weight", "regular"),
                font_path=font_path,
                rotation=region_spec.get("rotation", 0),
            )

            if text_img.size != (region_width, region_height):
                text_img = text_img.resize((region_width, region_height), Image.LANCZOS)

            combined_template.paste(text_img, (x1, y1))

        full_mask = self.extract_text_mask(np.array(combined_template))
        
        combined_latent = self.encode_image(combined_template)
        latent_list = self.compute_inversion_latents(combined_latent, noise, timesteps)

        # Downsample mask to latent space
        latent_h = 2 * (height // (self.vae_scale_factor * 2))
        latent_w = 2 * (width // (self.vae_scale_factor * 2))
        mask_area = cv2.resize(full_mask, (latent_w, latent_h), interpolation=cv2.INTER_AREA)
        mask_latent = (mask_area > 2).astype(np.float32)
        mask_latent = torch.from_numpy(mask_latent).unsqueeze(0).unsqueeze(0).to(self.device)

        return {
            "latent_list": latent_list,
            "mask_latent": mask_latent,
            "full_mask": full_mask,
            "combined_template": combined_template,
            "total_steps": len(timesteps),
        }

    def render_plan_template(
        self,
        typography_plan: dict,
        image_size: Tuple[int, int],
    ) -> Tuple[Image.Image, np.ndarray]:
        """Lightweight rendering: only returns (combined_template, full_mask), no VAE/inversion"""
        width, height = image_size
        combined_template = Image.new("RGB", (width, height), "black")

        for region_spec in typography_plan.get("text_regions", []):
            content = region_spec["content"]
            bbox = region_spec["bbox"]
            x1, y1 = int(bbox[0] * width), int(bbox[1] * height)
            x2, y2 = int(bbox[2] * width), int(bbox[3] * height)
            rw, rh = max(x2 - x1, 1), max(y2 - y1, 1)

            font_path = region_spec.get("font_path") or resolve_font_name(region_spec.get("font"))

            text_img = self.render_text_template(
                content, rw, rh,
                text_color=region_spec.get("color", "#FFFFFF"),
                force_latex=region_spec.get("is_latex", False),
                font_weight=region_spec.get("font_weight", "regular"),
                font_path=font_path,
                rotation=region_spec.get("rotation", 0),
            )
            if text_img.size != (rw, rh):
                text_img = text_img.resize((rw, rh), Image.LANCZOS)
            combined_template.paste(text_img, (x1, y1))

        full_mask = self.extract_text_mask(np.array(combined_template))
        return combined_template, full_mask

    def inject_latent(
        self,
        current_latent: torch.Tensor,
        injection_data: dict,
        step_idx: int,
        mask_strength: float = 1.0,
        timestep_ratio: Tuple[float, float] = (0.2, 0.8),
        freq_decompose: bool = True,
        freq_kernel_size: int = 5,
    ) -> torch.Tensor:
        """Inject text region latent into current latent (post-injection)
        
        Args:
            current_latent: latent after scheduler step
            injection_data: data returned by prepare_injection
            step_idx: latent index for injection (post-injection is denoising_step + 1)
            mask_strength: spatial blending strength (0-1)
            timestep_ratio: timestep injection range (t_start, t_end)
            freq_decompose: whether to enable frequency decomposition injection
            freq_kernel_size: Gaussian blur kernel size
        """
        total_steps = injection_data["total_steps"]
        
        # Determine if injection is needed at current step
        t_start, t_end = timestep_ratio
        step_ratio = step_idx / total_steps
        if not (t_start <= step_ratio < t_end):
            return current_latent
        
        latent_list = injection_data.get("latent_list")
        if latent_list is None:
            return current_latent
        
        idx = min(step_idx, len(latent_list) - 1)
        text_latent = latent_list[idx]
        mask = injection_data["mask_latent"]
        mask = mask.expand_as(current_latent)
        
        # Frequency decomposition injection
        if freq_decompose:
            return self._freq_decompose_inject(
                current_latent, text_latent, mask, mask_strength, freq_kernel_size
            )
        else:
            return current_latent * (1 - mask * mask_strength) + text_latent * mask * mask_strength


def create_glyph_injector(pipeline, device: str = "cuda") -> GlyphInjector:
    """Create GlyphInjector from pipeline"""
    return GlyphInjector(
        vae=pipeline.vae,
        scheduler=pipeline.scheduler,
        device=device,
        dtype=pipeline.vae.dtype,
    )
