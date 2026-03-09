"""
Prompt-Latent Attention Enhancement

在 attention logits 添加 log(scale) 偏置，增强 text token 与 glyph patch 之间的注意力。
"""

import math
import re
from typing import List, Optional

import torch
import torch.nn.functional as F

SEQ_MULTI_OF = 32


def find_quoted_token_indices(tokenizer, prompt: str, max_seq_length: int = 512) -> List[int]:
    """提取双引号内文本对应的 token 索引。"""
    matches = [m.group(1) for m in re.finditer(r'"([^"]*)"', prompt)]
    if not matches:
        return []

    messages = [{"role": "user", "content": prompt}]
    processed = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=True,
    )
    encoding = tokenizer(processed, padding="max_length", max_length=max_seq_length, truncation=True)
    full_ids = encoding.input_ids
    attn_mask = encoding.attention_mask
    num_real = sum(attn_mask)
    real_ids = full_ids[:num_real]

    result: list[int] = []
    for text in matches:
        sub_ids = tokenizer.encode(text, add_special_tokens=False)
        found = _find_subseq(real_ids, sub_ids)
        if found is None:
            sub_ids_q = tokenizer.encode(f'"{text}"', add_special_tokens=False)
            found = _find_subseq(real_ids, sub_ids_q)
        if found is not None:
            result.extend(found)

    return sorted(set(result))


def find_all_prompt_token_indices(tokenizer, prompt: str, max_seq_length: int = 512) -> List[int]:
    """返回 prompt 中所有 non-padding token 的索引。"""
    messages = [{"role": "user", "content": prompt}]
    processed = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=True,
    )
    encoding = tokenizer(processed, padding="max_length", max_length=max_seq_length, truncation=True)
    num_real = sum(encoding.attention_mask)
    return list(range(num_real))


def _find_subseq(seq: list, subseq: list) -> Optional[List[int]]:
    """在 seq 中查找 subseq 的首次出现，返回索引列表。"""
    n, m = len(seq), len(subseq)
    if m == 0:
        return []
    for i in range(n - m + 1):
        if seq[i : i + m] == subseq:
            return list(range(i, i + m))
    return None


def compute_glyph_patch_indices(mask_latent: torch.Tensor, patch_size: int = 2) -> List[int]:
    """将 latent-space glyph mask 映射到 image patch 索引。"""
    mask = mask_latent[0, 0]
    H, W = mask.shape
    Hp, Wp = H // patch_size, W // patch_size

    mask_grid = mask[: Hp * patch_size, : Wp * patch_size].reshape(
        Hp, patch_size, Wp, patch_size
    )
    has_glyph = mask_grid.amax(dim=(1, 3)) > 0
    return torch.where(has_glyph.flatten())[0].tolist()


class _EnhancementState:
    """在 transformer layers 间共享的增强状态。"""

    def __init__(
        self,
        config,
        text_indices: List[int],
        image_indices: List[int],
        x_seq_len: int,
        cap_seq_len: int,
        num_layers: int,
        logger=None,
        Hp: Optional[int] = None,
        Wp: Optional[int] = None,
    ):
        self.config = config
        self.text_indices = text_indices
        self.image_indices = image_indices
        self.x_seq_len = x_seq_len
        self.cap_seq_len = cap_seq_len
        self.num_layers = num_layers

        self.current_step = 0
        self.total_steps = 1
        self._bias_cache: dict = {}

        # Attention heatmap visualization
        self.logger = logger
        self.Hp = Hp
        self.Wp = Wp
        self._vis_layer = num_layers // 2

    def should_enhance(self, layer_idx: int) -> bool:
        if not self.text_indices or not self.image_indices:
            return False
        step_ratio = self.current_step / max(self.total_steps, 1)
        t_start, t_end = self.config.timestep_ratio
        if not (t_start <= step_ratio < t_end):
            return False
        layers = self.config.attn_enhance_layers
        if layers is not None and layer_idx not in layers:
            return False
        return True

    def get_bias(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """构造 (1, 1, L, L) 的 additive attention bias。"""
        key = (seq_len, device)
        if key in self._bias_cache:
            return self._bias_cache[key].to(dtype)

        bias = torch.zeros(1, 1, seq_len, seq_len, device=device, dtype=torch.float32)
        x_len = self.x_seq_len
        log_scale = math.log(max(self.config.attn_enhance_scale, 1e-6))

        text_abs = torch.tensor(
            [x_len + i for i in self.text_indices if i < self.cap_seq_len],
            dtype=torch.long, device=device,
        )
        image_abs = torch.tensor(
            [i for i in self.image_indices if i < x_len],
            dtype=torch.long, device=device,
        )

        if len(text_abs) == 0 or len(image_abs) == 0:
            self._bias_cache[key] = bias
            return bias.to(dtype)

        if self.config.attn_enhance_image_to_text:
            bias[0, 0, image_abs.unsqueeze(1), text_abs.unsqueeze(0)] = log_scale

        if self.config.attn_enhance_text_to_image:
            bias[0, 0, text_abs.unsqueeze(1), image_abs.unsqueeze(0)] = log_scale

        suppress = self.config.attn_suppress_scale
        if suppress < 1.0 and suppress > 0:
            log_suppress = math.log(suppress)
            all_image = torch.arange(x_len, dtype=torch.long, device=device)
            glyph_set = set(self.image_indices)
            non_glyph = torch.tensor(
                [i for i in range(x_len) if i not in glyph_set],
                dtype=torch.long, device=device,
            )
            if len(non_glyph) > 0:
                bias[0, 0, non_glyph.unsqueeze(1), text_abs.unsqueeze(0)] = log_suppress
                bias[0, 0, text_abs.unsqueeze(1), non_glyph.unsqueeze(0)] = log_suppress

        self._bias_cache[key] = bias
        return bias.to(dtype)


class EnhancedAttnProcessor:
    """包装原始 attention processor，注入 logit-bias 增强注意力。"""

    def __init__(self, original, layer_idx: int, state: _EnhancementState):
        object.__setattr__(self, "_original", original)
        object.__setattr__(self, "_layer_idx", layer_idx)
        object.__setattr__(self, "_state", state)

    def __getattr__(self, name):
        return getattr(self._original, name)

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states=None,
        attention_mask=None,
        freqs_cis=None,
        **kwargs,
    ) -> torch.Tensor:
        if not self._state.should_enhance(self._layer_idx):
            return self._original(
                attn, hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
                freqs_cis=freqs_cis,
                **kwargs,
            )

        return self._forward_with_bias(
            attn, hidden_states, attention_mask, freqs_cis
        )

    def _forward_with_bias(
        self, attn, hidden_states, attention_mask, freqs_cis
    ) -> torch.Tensor:
        """手动 forward 并注入 bias。"""
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        query = query.unflatten(-1, (attn.heads, -1))
        key = key.unflatten(-1, (attn.heads, -1))
        value = value.unflatten(-1, (attn.heads, -1))

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        if freqs_cis is not None:
            def _rope(x_in, fc):
                with torch.amp.autocast("cuda", enabled=False):
                    x = torch.view_as_complex(x_in.float().reshape(*x_in.shape[:-1], -1, 2))
                    fc = fc.unsqueeze(2)
                    return torch.view_as_real(x * fc).flatten(3).type_as(x_in)
            query = _rope(query, freqs_cis)
            key = _rope(key, freqs_cis)

        dtype = query.dtype
        query, key = query.to(dtype), key.to(dtype)

        B, N, H, D = query.shape
        q = query.transpose(1, 2)
        k = key.transpose(1, 2)
        v = value.transpose(1, 2).to(dtype)

        # 注入 bias
        attn_bias = self._state.get_bias(N, q.device, q.dtype)

        if attention_mask is not None:
            if attention_mask.ndim == 2:
                attention_mask = attention_mask[:, None, None, :]
            pad_bias = torch.zeros_like(attention_mask, dtype=q.dtype)
            pad_bias.masked_fill_(~attention_mask.bool(), float("-inf"))
            attn_bias = attn_bias + pad_bias

        # Save attention heatmap for the visualization layer (debug only)
        if (self._state.config.debug
                and self._state.logger is not None
                and self._state.Hp is not None
                and self._layer_idx == self._state._vis_layer):
            self._save_attn_heatmap(q, k, attn_bias)

        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_bias, dropout_p=0.0, is_causal=False
        )

        hidden_states = out.transpose(1, 2).flatten(2, 3).to(dtype)
        output = attn.to_out[0](hidden_states)
        if len(attn.to_out) > 1:
            output = attn.to_out[1](output)
        return output

    @torch.no_grad()
    def _save_attn_heatmap(self, q, k, attn_bias):
        """计算 text→image attention 并保存灰度热力图。

        只提取 text query 行，避免计算完整 N×N attention 矩阵。
        """
        import numpy as np
        from PIL import Image

        state = self._state
        x_len = state.x_seq_len
        num_patches = state.Hp * state.Wp

        text_abs = torch.tensor(
            [x_len + i for i in state.text_indices if i < state.cap_seq_len],
            dtype=torch.long, device=q.device,
        )
        img_abs = torch.arange(num_patches, dtype=torch.long, device=q.device)

        if len(text_abs) == 0 or len(img_abs) == 0:
            return

        scale = q.shape[-1] ** -0.5
        q_text = q[:, :, text_abs, :]                                   # (B, H, T, D)
        logits = torch.matmul(q_text * scale, k.transpose(-2, -1))     # (B, H, T, N)

        if attn_bias is not None:
            logits = logits + attn_bias[:, :, text_abs, :]

        attn_weights = logits.float().softmax(dim=-1)                   # (B, H, T, N)
        text_to_img = attn_weights[:, :, :, img_abs]                    # (B, H, T, I)
        heatmap = text_to_img.mean(dim=(0, 1, 2))                      # (I,)
        heatmap = heatmap.reshape(state.Hp, state.Wp)

        h_min, h_max = heatmap.min(), heatmap.max()
        if h_max > h_min:
            heatmap = (heatmap - h_min) / (h_max - h_min)
        else:
            heatmap.zero_()

        heatmap_np = (heatmap.cpu().numpy() * 255).astype(np.uint8)

        vis_h = max(state.Hp * 8, 256)
        vis_w = max(state.Wp * 8, 256)
        heatmap_pil = Image.fromarray(heatmap_np, mode="L").resize(
            (vis_w, vis_h), Image.BILINEAR,
        )

        step = state.current_step
        state.logger.save_image(
            heatmap_pil.convert("RGB"),
            f"attn_text2img_step{step:02d}_L{self._layer_idx:02d}",
            caption=f"text→image attn | step={step}/{state.total_steps} | layer={self._layer_idx}",
            subfolder="attention",
        )


class AttentionEnhancement:
    """Attention Enhancement 管理器。"""

    def __init__(self, state: _EnhancementState):
        self._state = state
        self._installed = False

    @classmethod
    def create(
        cls,
        config,
        tokenizer,
        prompt: str,
        mask_latent: torch.Tensor,
        latent_height: int,
        latent_width: int,
        cap_ori_len: int,
        num_layers: int,
        patch_size: int = 2,
        max_seq_length: int = 512,
        logger=None,
    ) -> Optional["AttentionEnhancement"]:
        """工厂方法：创建 enhancement。"""
        if not config.attn_enhance_enabled:
            return None

        text_indices = find_quoted_token_indices(tokenizer, prompt, max_seq_length)
        if not text_indices:
            text_indices = find_all_prompt_token_indices(tokenizer, prompt, max_seq_length)
            if logger:
                logger.info("[AttnEnhancement] fallback 到整个 prompt 序列")

        image_indices = compute_glyph_patch_indices(mask_latent, patch_size)

        if not text_indices or not image_indices:
            print(f"[AttnEnhancement] 跳过：text_tokens={len(text_indices)}, glyph_patches={len(image_indices)}")
            return None

        Hp, Wp = latent_height // patch_size, latent_width // patch_size
        num_patches = Hp * Wp
        x_seq_len = num_patches + (-num_patches) % SEQ_MULTI_OF
        cap_seq_len = cap_ori_len + (-cap_ori_len) % SEQ_MULTI_OF

        state = _EnhancementState(
            config, text_indices, image_indices, x_seq_len, cap_seq_len, num_layers,
            logger=logger, Hp=Hp, Wp=Wp,
        )

        t_start, t_end = config.timestep_ratio
        print(f"[AttnEnhancement] 激活：text={len(text_indices)}, patches={len(image_indices)}, "
              f"scale={config.attn_enhance_scale:.1f}, timestep=({t_start:.0%}, {t_end:.0%})")

        if logger:
            logger.info(f"[AttnEnhancement] text_tokens={len(text_indices)}, glyph_patches={len(image_indices)}")

        return cls(state)

    def install(self, transformer) -> None:
        """替换所有层的 attention processor。"""
        if self._installed:
            return
        for idx, layer in enumerate(transformer.layers):
            original = layer.attention.processor
            layer.attention.processor = EnhancedAttnProcessor(original, idx, self._state)
        self._installed = True

    def uninstall(self, transformer) -> None:
        """还原所有 attention processor。"""
        if not self._installed:
            return
        for layer in transformer.layers:
            proc = layer.attention.processor
            if isinstance(proc, EnhancedAttnProcessor):
                layer.attention.processor = proc._original
        self._installed = False

    def set_step(self, step: int, total_steps: int) -> None:
        """更新当前去噪步。"""
        self._state.current_step = step
        self._state.total_steps = total_steps


