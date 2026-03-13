import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from diffusers.models.attention import Attention
from diffusers.models.attention_dispatch import dispatch_attention_fn
from diffusers.models.normalization import RMSNorm


class IPAZImageAttnProcessor:
    """
    Z-Image attention processor with IP-Adapter support.
    """

    _attention_backend = None
    _parallel_config = None

    def __init__(self, hidden_size, cross_attention_dim=None, scale=1.0):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "IPAZImageAttnProcessor requires PyTorch 2.0. "
                "To use it, please upgrade PyTorch to version 2.0 or higher."
            )

        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim
        self.scale = scale

        self.to_k_ip = nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False)
        self.to_v_ip = nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False)
        self._norm_added_k = None
        self._norm_added_k_dim = None

    def _get_norm_added_k(self, head_dim: int):
        if self._norm_added_k is None or self._norm_added_k_dim != head_dim:
            self._norm_added_k = RMSNorm(head_dim, eps=1e-5, elementwise_affine=False)
            self._norm_added_k_dim = head_dim
        return self._norm_added_k

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        freqs_cis: Optional[torch.Tensor] = None,
        image_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        query = query.unflatten(-1, (attn.heads, -1))
        key = key.unflatten(-1, (attn.heads, -1))
        value = value.unflatten(-1, (attn.heads, -1))

        # Apply Norms
        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # Apply RoPE
        def apply_rotary_emb(x_in: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
            with torch.amp.autocast("cuda", enabled=False):
                x = torch.view_as_complex(x_in.float().reshape(*x_in.shape[:-1], -1, 2))
                freqs_cis = freqs_cis.unsqueeze(2)
                x_out = torch.view_as_real(x * freqs_cis).flatten(3)
                return x_out.type_as(x_in)

        if freqs_cis is not None:
            query = apply_rotary_emb(query, freqs_cis)
            key = apply_rotary_emb(key, freqs_cis)

        # Cast to correct dtype
        dtype = query.dtype
        query, key = query.to(dtype), key.to(dtype)

        # From [batch, seq_len] to [batch, 1, 1, seq_len]
        if attention_mask is not None and attention_mask.ndim == 2:
            attention_mask = attention_mask[:, None, None, :]

        # Main attention
        hidden_states = dispatch_attention_fn(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
            backend=self._attention_backend,
            parallel_config=self._parallel_config,
        )

        # IP-Adapter attention
        if image_emb is not None:
            ip_key = self.to_k_ip(image_emb)
            ip_value = self.to_v_ip(image_emb)

            ip_key = ip_key.unflatten(-1, (attn.heads, -1))
            ip_value = ip_value.unflatten(-1, (attn.heads, -1))

            head_dim = ip_key.shape[-1]
            norm_added_k = self._get_norm_added_k(head_dim).to(ip_key.device)
            ip_key = norm_added_k(ip_key)

            ip_hidden_states = dispatch_attention_fn(
                query,
                ip_key,
                ip_value,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=False,
                backend=self._attention_backend,
                parallel_config=self._parallel_config,
            )
            hidden_states = hidden_states + self.scale * ip_hidden_states

        # Reshape back
        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.to(dtype)

        output = attn.to_out[0](hidden_states)
        if len(attn.to_out) > 1:  # dropout
            output = attn.to_out[1](output)

        return output
