import torch
from typing import Union, List, Optional
from PIL import Image

def crop_resize_and_pad(image_path, bbox, target_size=(512, 512)):
    """
    根据给定的 bbox 从图片中抠出对应区域，按比例调整到目标尺寸，填充剩余空白为黑色。
    """
    image = Image.open(image_path).convert("RGB")

    x_coords = [point[0] for point in bbox]
    y_coords = [point[1] for point in bbox]
    left = int(min(x_coords))
    top = int(min(y_coords))
    right = int(max(x_coords))
    bottom = int(max(y_coords))

    cropped_image = image.crop((left, top, right, bottom))
    cropped_width, cropped_height = cropped_image.size
    
    target_width, target_height = target_size
    scale = min(target_width / cropped_width, target_height / cropped_height)
    new_width = int(cropped_width * scale)
    new_height = int(cropped_height * scale)

    resized_image = cropped_image.resize((new_width, new_height), Image.Resampling.BILINEAR)

    padded_image = Image.new("RGB", target_size, (0, 0, 0))
    left_padding = (target_width - new_width) // 2
    top_padding = (target_height - new_height) // 2
    padded_image.paste(resized_image, (left_padding, top_padding))
    
    return padded_image

def retrieve_latents(encoder_output: torch.Tensor, sample_mode: str = "sample"):
    if hasattr(encoder_output, "latent_dist"):
        if sample_mode == "sample":
            return encoder_output.latent_dist.sample()
        elif sample_mode == "argmax":
            return encoder_output.latent_dist.mode()
    elif hasattr(encoder_output, "latents"):
        return encoder_output.latents
    raise AttributeError("Could not access latents of provided encoder_output")

def prepare_mask_latents4training(
    mask, masked_image, batch_size, num_channels_latents, height, width, dtype, device, vae, vae_scale_factor
):
    height = 2 * (int(height) // (vae_scale_factor * 2))
    width = 2 * (int(width) // (vae_scale_factor * 2))

    if masked_image.shape[1] == num_channels_latents:
        masked_image_latents = masked_image
    else:
        masked_image_latents = retrieve_latents(vae.encode(masked_image))

    masked_image_latents = (masked_image_latents - vae.config.shift_factor) * vae.config.scaling_factor
    masked_image_latents = masked_image_latents.to(device=device, dtype=dtype)

    if mask.shape[0] < batch_size:
        mask = mask.repeat(batch_size // mask.shape[0], 1, 1, 1)
    if masked_image_latents.shape[0] < batch_size:
        masked_image_latents = masked_image_latents.repeat(batch_size // masked_image_latents.shape[0], 1, 1, 1)

    masked_image_latents = pack_latents(masked_image_latents)

    mask = mask[:, 0, :, :]
    mask = mask.view(batch_size, height, vae_scale_factor, width, vae_scale_factor)
    mask = mask.permute(0, 2, 4, 1, 3)
    mask = mask.reshape(batch_size, vae_scale_factor * vae_scale_factor, height, width)
    
    # Pack the mask
    mask_latents = pack_latents(mask)
    mask_latents = mask_latents.to(device=device, dtype=dtype)

    return mask_latents, masked_image_latents

def get_t5_prompt_embeds(
    text_encoder_2,
    tokenizer_2,
    prompt: Union[str, List[str]],
    num_images_per_prompt: int = 1,
    max_sequence_length: int = 512,
    device: Optional[torch.device] = None,
):
    text_inputs = tokenizer_2(
        prompt,
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids.to(device)
    prompt_embeds = text_encoder_2(text_input_ids, output_hidden_states=False)[0]
    
    dtype = text_encoder_2.dtype
    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
    
    bs, seq_len, _ = prompt_embeds.shape
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(bs * num_images_per_prompt, seq_len, -1)
    
    return prompt_embeds

def get_clip_prompt_embeds(
    text_encoder,
    tokenizer,
    prompt: Union[str, List[str]],
    num_images_per_prompt: int = 1,
    device: Optional[torch.device] = None,
):
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids.to(device)
    prompt_embeds = text_encoder(text_input_ids, output_hidden_states=False).pooler_output
    prompt_embeds = prompt_embeds.to(dtype=text_encoder.dtype, device=device)
    
    bs = len(prompt)
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(bs * num_images_per_prompt, -1)

    return prompt_embeds

def encode_prompt(text_encoders, tokenizers, prompt, device, max_sequence_length=512):
    pooled_prompt_embeds = get_clip_prompt_embeds(
        text_encoders[0], tokenizers[0], prompt, device=device
    )
    prompt_embeds = get_t5_prompt_embeds(
        text_encoders[1], tokenizers[1], prompt, max_sequence_length=max_sequence_length, device=device
    )
    return prompt_embeds, pooled_prompt_embeds

def pack_latents(latents):
    batch_size, num_channels, height, width = latents.shape
    latents = latents.view(batch_size, num_channels, height // 2, 2, width // 2, 2)
    latents = latents.permute(0, 2, 4, 1, 3, 5)
    latents = latents.reshape(batch_size, (height // 2) * (width // 2), num_channels * 4)
    return latents

def unpack_latents(latents, height, width, vae_scale_factor):
    batch_size, num_patches, channels = latents.shape
    height = height // vae_scale_factor
    width = width // vae_scale_factor
    latents = latents.view(batch_size, height, width, channels // 4, 2, 2)
    latents = latents.permute(0, 3, 1, 4, 2, 5)
    latents = latents.reshape(batch_size, channels // (2 * 2), height * 2, width * 2)
    return latents

def prepare_latent_image_ids(height, width, device, dtype):
    latent_image_ids = torch.zeros(height // 2, width // 2, 3, device=device, dtype=dtype)
    latent_image_ids[..., 1] += torch.arange(height // 2, device=device, dtype=dtype)[:, None]
    latent_image_ids[..., 2] += torch.arange(width // 2, device=device, dtype=dtype)[None, :]
    return latent_image_ids.view(-1, 3)

def get_sigmas(timesteps, noise_scheduler_copy, n_dim=4, dtype=torch.float32, device='cpu'):
    sigmas = noise_scheduler_copy.sigmas.to(device=device, dtype=dtype)
    schedule_timesteps = noise_scheduler_copy.timesteps.to(device)
    timesteps = timesteps.to(device)
    step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]
    sigma = sigmas[step_indices].flatten()
    while len(sigma.shape) < n_dim:
        sigma = sigma.unsqueeze(-1)
    return sigma
