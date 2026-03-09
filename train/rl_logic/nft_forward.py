"""
Helper functions for NFT forward passes
"""

import torch
from typing import Tuple


def forward_with_old_model(
    transformer,
    image_proj_model,
    latents,
    timesteps,
    prompt_embeds,
    pooled_prompt_embeds,
    ip_tokens,
    text_ids,
    img_ids,
    guidance,
    mask_latents,
    masked_image_latents,
    use_old: bool = False,
):
    """
    Forward pass that can use either the current or old IP-Adapter parameters.
    
    For NFT training, we need to:
    1. Get predictions from the current model
    2. Get predictions from the old model (stored separately)
    3. Get predictions from the frozen reference model (base transformer without IP)
    
    Args:
        transformer: The transformer model
        image_proj_model: The image projection model
        latents: Noisy latents
        timesteps: Timestep values
        prompt_embeds: Text prompt embeddings
        pooled_prompt_embeds: Pooled text embeddings
        ip_tokens: IP-Adapter tokens (can be from current or old image_proj_model)
        text_ids: Text position IDs
        img_ids: Image position IDs
        guidance: Guidance scale values
        mask_latents: Mask latents for inpainting
        masked_image_latents: Masked image latents for inpainting
        use_old: Whether to use old parameters (for NFT)
        
    Returns:
        Model prediction
    """
    from train.flux_ip.utils import pack_latents, unpack_latents
    
    b, c, h, w = latents.shape
    
    # Pack latents for transformer input
    packed_noisy_latents = pack_latents(latents, b, c, h, w)
    packed_masked_image_latents = pack_latents(masked_image_latents, b, c, h, w)
    
    # Pack mask (assume vae_scale_factor=8)
    vae_scale_factor = 8
    mask_c = vae_scale_factor ** 2
    packed_mask = pack_latents(mask_latents.repeat(1, mask_c, 1, 1), b, mask_c, h, w)
    
    # Concatenate all inputs
    transformer_hidden_states = torch.cat(
        [packed_noisy_latents, packed_masked_image_latents, packed_mask], dim=-1
    )
    
    # Forward pass
    model_pred = transformer(
        hidden_states=transformer_hidden_states,
        timestep=timesteps / 1000,
        guidance=guidance,
        pooled_projections=pooled_prompt_embeds,
        encoder_hidden_states=prompt_embeds,
        image_emb=ip_tokens,
        txt_ids=text_ids,
        img_ids=img_ids,
        return_dict=False,
    )[0]
    
    # Unpack latents
    model_pred = unpack_latents(model_pred, h * 8, w * 8, 16)
    
    return model_pred


def copy_ip_adapter_params(
    src_image_proj,
    src_attn_processors,
    dst_image_proj,
    dst_attn_processors,
):
    """
    Copy IP-Adapter parameters from source to destination.
    This is used to create an "old" copy of the model for NFT training.
    
    Args:
        src_image_proj: Source image projection model
        src_attn_processors: Source attention processors
        dst_image_proj: Destination image projection model
        dst_attn_processors: Destination attention processors
    """
    with torch.no_grad():
        # Copy image projection model parameters
        for src_param, dst_param in zip(src_image_proj.parameters(), dst_image_proj.parameters()):
            dst_param.data.copy_(src_param.data)
        
        # Copy attention processor parameters
        src_list = list(src_attn_processors.values())
        dst_list = list(dst_attn_processors.values())
        
        for src_proc, dst_proc in zip(src_list, dst_list):
            for src_param, dst_param in zip(src_proc.parameters(), dst_proc.parameters()):
                dst_param.data.copy_(src_param.data)

