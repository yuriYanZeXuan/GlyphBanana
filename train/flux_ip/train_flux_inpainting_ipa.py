#!/usr/bin/env python
# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team and The InstantX Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and

import argparse
import copy
import gc
import itertools
import logging
import math
import os
import random
import shutil
import warnings
from contextlib import nullcontext
from pathlib import Path

from PIL import Image

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import numpy as np
import torch
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration, set_seed
from huggingface_hub import create_repo, upload_folder
from huggingface_hub.utils import insecure_hashlib
from datasets import load_dataset
from PIL import Image, ImageDraw
from PIL.ImageOps import exif_transpose
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms.functional import crop
from tqdm.auto import tqdm
from transformers import CLIPTextModelWithProjection, CLIPTokenizer, PretrainedConfig, T5EncoderModel, T5TokenizerFast, CLIPTextModel

import diffusers
from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
)
from diffusers.optimization import get_scheduler
from diffusers.training_utils import compute_density_for_timestep_sampling, compute_loss_weighting_for_sd3
from diffusers.utils import (
    check_min_version,
    is_wandb_available,
)
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
from diffusers.utils.torch_utils import is_compiled_module

from diffusers.models.controlnet_flux import FluxControlNetModel
from transformer_flux_inpainting import FluxTransformerBlock, FluxSingleTransformerBlock, FluxTransformer2DModel

from typing import Any, Callable, Dict, List, Optional, Union
from datetime import timedelta
from accelerate.utils import InitProcessGroupKwargs
from ocr import crop_resize_and_pad 

from torchvision.transforms.functional import to_pil_image, to_tensor


if is_wandb_available():
    import wandb

import torch.nn as nn
from attention_processor import IPAFluxAttnProcessor2_0
from transformers import AutoProcessor, SiglipVisionModel
from pipeline_flux_inpainting import FluxFillPipeline

import warnings
warnings.filterwarnings('ignore')
    
# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.30.0.dev0")

logger = get_logger(__name__)


def load_text_encoders(class_one, class_two):
    text_encoder_one = class_one.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision, variant=args.variant
    )
    text_encoder_two = class_two.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder_2", revision=args.revision, variant=args.variant
    )
    return text_encoder_one, text_encoder_two


def import_model_class_from_model_name_or_path(
    pretrained_model_name_or_path: str, revision: str, subfolder: str = "text_encoder"
):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path, subfolder=subfolder, revision=revision
    )
    model_class = text_encoder_config.architectures[0]
    if model_class == "CLIPTextModelWithProjection":
        from transformers import CLIPTextModelWithProjection

        return CLIPTextModelWithProjection
    elif model_class == "T5EncoderModel":
        from transformers import T5EncoderModel

        return T5EncoderModel
    
    elif model_class == "CLIPTextModel":
        from transformers import CLIPTextModel
        
        return CLIPTextModel
    else:
        raise ValueError(f"{model_class} is not supported.")


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--ip_adapter_path",
        type=str,
        default=None,
        required=False,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help=(
            "The name of the Dataset (from the HuggingFace hub) containing the training data of instance images (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that 🤗 Datasets can understand."
        ),
    )
    parser.add_argument(
        "--siglip_path",
        type=str,
        default=None,
        help=(
            "google siglip path ."
        ),
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The config of the Dataset, leave as None if there's only one config.",
    )
    
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
        help=(
            "A folder containing the training data. Folder contents must follow the structure described in"
            " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
            " must exist to provide the captions for the images. Ignored if `dataset_name` is specified."
        ),
    )
    
    parser.add_argument(
        "--train_data_json",
        type=str,
        default=None,
        help=(
            "A folder containing the training data. Folder contents must follow the structure described in"
            " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
            " must exist to provide the captions for the images. Ignored if `dataset_name` is specified."
        ),
    )

    parser.add_argument(
        "--init_ckpt",
        type=str,
        default=None,
        help=(
            "A folder containing the training data. Folder contents must follow the structure described in"
            " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
            " must exist to provide the captions for the images. Ignored if `dataset_name` is specified."
        ),
    )

    parser.add_argument(
        "--image_column",
        type=str,
        default="image",
        help="The column of the dataset containing the target image. By "
        "default, the standard Image Dataset maps out 'file_name' "
        "to 'image'.",
    )
    parser.add_argument(
        "--caption_column",
        type=str,
        default="text",
        help="The column of the dataset containing the instance prompt for each image",
    )
    parser.add_argument(
        "--short_caption_column",
        type=str,
        default="text",
        help="The column of the dataset containing the instance prompt for each image",
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    parser.add_argument("--repeats", type=int, default=1, help="How many times to repeat the training data.")

    parser.add_argument(
        "--max_sequence_length",
        type=int,
        default=256,
        help="Maximum sequence length to use with with the T5 text encoder",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="flux-controlnet",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=1024,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--center_crop",
        default=False,
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )
    parser.add_argument(
        "--random_flip",
        action="store_true",
        help="whether to randomly flip images horizontally",
    )
    parser.add_argument(
        "--train_text_encoder",
        action="store_true",
        help="Whether to train the text encoder. If set, the text encoder should be float32 precision.",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints can be used both as final"
            " checkpoints in case they are better than the last checkpoint, and are also suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )

    parser.add_argument(
        "--text_encoder_lr",
        type=float,
        default=5e-6,
        help="Text encoder learning rate to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--weighting_scheme",
        type=str,
        default="logit_normal",
        choices=["sigma_sqrt", "logit_normal", "mode", "cosmap"],
    )
    parser.add_argument(
        "--logit_mean", type=float, default=0.0, help="mean to use when using the `'logit_normal'` weighting scheme."
    )
    parser.add_argument(
        "--logit_std", type=float, default=1.0, help="std to use when using the `'logit_normal'` weighting scheme."
    )
    parser.add_argument(
        "--mode_scale",
        type=float,
        default=1.29,
        help="Scale of mode weighting scheme. Only effective when using the `'mode'` as the `weighting_scheme`.",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="AdamW",
        help=('The optimizer type to use. Choose between ["AdamW", "prodigy"]'),
    )

    parser.add_argument(
        "--use_8bit_adam",
        action="store_true",
        help="Whether or not to use 8-bit Adam from bitsandbytes. Ignored if optimizer is not set to AdamW",
    )

    parser.add_argument(
        "--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam and Prodigy optimizers."
    )
    parser.add_argument(
        "--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam and Prodigy optimizers."
    )
    parser.add_argument(
        "--prodigy_beta3",
        type=float,
        default=None,
        help="coefficients for computing the Prodidy stepsize using running averages. If set to None, "
        "uses the value of square root of beta2. Ignored if optimizer is adamW",
    )
    parser.add_argument("--prodigy_decouple", type=bool, default=True, help="Use AdamW style decoupled weight decay")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-04, help="Weight decay to use for unet params")
    parser.add_argument(
        "--adam_weight_decay_text_encoder", type=float, default=1e-03, help="Weight decay to use for text_encoder"
    )

    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-08,
        help="Epsilon value for the Adam optimizer and Prodigy optimizers.",
    )

    parser.add_argument(
        "--prodigy_use_bias_correction",
        type=bool,
        default=True,
        help="Turn on Adam's bias correction. True by default. Ignored if optimizer is adamW",
    )
    parser.add_argument(
        "--prodigy_safeguard_warmup",
        type=bool,
        default=True,
        help="Remove lr from the denominator of D estimate to avoid issues during warm-up stage. True by default. "
        "Ignored if optimizer is adamW",
    )
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--allow_adam_bf16",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args


def get_t5_prompt_embeds(
    text_encoder_2,
    tokenizer_2,
    prompt: Union[str, List[str]] = None,
    num_images_per_prompt: int = 1,
    max_sequence_length: int = 512,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
):
    device = device
    dtype = dtype or text_encoder_2.dtype

    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    text_inputs = tokenizer_2(
        prompt,
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        return_length=False,
        return_overflowing_tokens=False,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids
    untruncated_ids = tokenizer_2(prompt, padding="longest", return_tensors="pt").input_ids

    if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
        removed_text = tokenizer_2.batch_decode(untruncated_ids[:, tokenizer_2.model_max_length - 1 : -1])
    prompt_embeds = text_encoder_2(text_input_ids.to(device), output_hidden_states=False)[0]

    dtype = text_encoder_2.dtype
    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

    _, seq_len, _ = prompt_embeds.shape

    # duplicate text embeddings and attention mask for each generation per prompt, using mps friendly method
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

    return prompt_embeds

def retrieve_latents(
        encoder_output: torch.Tensor, generator: Optional[torch.Generator] = None, sample_mode: str = "sample"
    ):
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        return encoder_output.latent_dist.sample(generator)
    elif hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        return encoder_output.latent_dist.mode()
    elif hasattr(encoder_output, "latents"):
        return encoder_output.latents
    else:
        raise AttributeError("Could not access latents of provided encoder_output")

def prepare_mask_latents4training(
    mask,
    masked_image,
    batch_size,
    num_channels_latents,
    num_images_per_prompt,
    height,
    width,
    dtype,
    device,
    generator,
    vae,
    vae_scale_factor,
):
    """
    Prepares the mask and masked image latents for the pipeline.

    Args:
        mask (`torch.Tensor`): The mask tensor. Shape: `(B, 1, H, W)`.
        masked_image (`torch.Tensor`): The masked image tensor. Shape: `(B, C, H, W)`.
        batch_size (`int`): The batch size.
        num_channels_latents (`int`): The number of latent channels.
        num_images_per_prompt (`int`): The number of images to generate per prompt.
        height (`int`): The height of the image.
        width (`int`): The width of the image.
        dtype (`torch.dtype`): The data type for the tensors.
        device (`torch.device`): The device to use for computations.
        generator (`torch.Generator`): The random generator for sampling.
        vae (`AutoencoderKL`): The Variational Auto-Encoder model.
        vae_scale_factor (`int`): The scale factor of the VAE.

    Returns:
        Tuple[`torch.Tensor`, `torch.Tensor`]: The processed mask and masked image latents.
    """
    # 1. Calculate the height and width of the latents
    # VAE applies 8x compression on images, and the latent height and width must be divisible by 2.
    height = 2 * (int(height) // (vae_scale_factor * 2))
    width = 2 * (int(width) // (vae_scale_factor * 2))

    # 2. Encode the masked image
    if masked_image.shape[1] == num_channels_latents:
        masked_image_latents = masked_image
    else:
        masked_image_latents = retrieve_latents(vae.encode(masked_image), generator=generator)

    masked_image_latents = (masked_image_latents - vae.config.shift_factor) * vae.config.scaling_factor
    masked_image_latents = masked_image_latents.to(device=device, dtype=dtype)

    # 3. Duplicate mask and masked_image_latents for each generation per prompt
    batch_size = batch_size * num_images_per_prompt
    if mask.shape[0] < batch_size:
        if not batch_size % mask.shape[0] == 0:
            raise ValueError(
                "The passed mask and the required batch size don't match. Masks are supposed to be duplicated to"
                f" a total batch size of {batch_size}, but {mask.shape[0]} masks were passed. Make sure the number"
                " of masks that you pass is divisible by the total requested batch size."
            )
        mask = mask.repeat(batch_size // mask.shape[0], 1, 1, 1)
    if masked_image_latents.shape[0] < batch_size:
        if not batch_size % masked_image_latents.shape[0] == 0:
            raise ValueError(
                "The passed images and the required batch size don't match. Images are supposed to be duplicated"
                f" to a total batch size of {batch_size}, but {masked_image_latents.shape[0]} images were passed."
                " Make sure the number of images that you pass is divisible by the total requested batch size."
            )
        masked_image_latents = masked_image_latents.repeat(batch_size // masked_image_latents.shape[0], 1, 1, 1)

    # 4. Pack the masked_image_latents
    # batch_size, num_channels_latents, height, width -> batch_size, height//2 * width//2 , num_channels_latents*4
    def fill_pack_latents(latents, batch_size, num_channels_latents, height, width):
        latents = latents.view(batch_size, num_channels_latents, height // 2, 2, width // 2, 2)
        latents = latents.permute(0, 2, 4, 1, 3, 5)
        latents = latents.reshape(batch_size, (height // 2) * (width // 2), num_channels_latents * 4)
        return latents

    # input torch.Size([2, 16, 64, 64]) 
    masked_image_latents = fill_pack_latents(
        masked_image_latents,  
        batch_size,
        num_channels_latents,
        height,
        width,
    )

    # 5. Resize mask to latents shape for concatenation
    mask = mask[:, 0, :, :]  # batch_size, 8 * height, 8 * width (mask has not been 8x compressed)
    mask = mask.view(
        batch_size, height, vae_scale_factor, width, vae_scale_factor
    )  # batch_size, height, 8, width, 8
    mask = mask.permute(0, 2, 4, 1, 3)  # batch_size, 8, 8, height, width
    mask = mask.reshape(
        batch_size, vae_scale_factor * vae_scale_factor, height, width
    )  # batch_size, 8*8, height, width

    # 6. Pack the mask
    # batch_size, 64, height, width -> batch_size, height//2 * width//2 , 64*2*2
    mask = pack_latents(
        mask,
        batch_size,
        vae_scale_factor * vae_scale_factor,
        height,
        width,
    )
    mask = mask.to(device=device, dtype=dtype)

    return mask, masked_image_latents
    

def get_clip_prompt_embeds(
    text_encoder,
    tokenizer,
    prompt: Union[str, List[str]],
    num_images_per_prompt: int = 1,
    device: Optional[torch.device] = None,
):
    device = device

    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_overflowing_tokens=False,
        return_length=False,
        return_tensors="pt",
    )

    text_input_ids = text_inputs.input_ids
    untruncated_ids = tokenizer(prompt, padding="longest", return_tensors="pt").input_ids
    if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
        removed_text = tokenizer.batch_decode(untruncated_ids[:, tokenizer.model_max_length - 1 : -1])
    prompt_embeds = text_encoder(text_input_ids.to(device), output_hidden_states=False)

    # Use pooled output of CLIPTextModel
    prompt_embeds = prompt_embeds.pooler_output
    prompt_embeds = prompt_embeds.to(dtype=text_encoder.dtype, device=device)

    # duplicate text embeddings for each generation per prompt, using mps friendly method
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, -1)

    return prompt_embeds

def encode_prompt(
    text_encoders,
    tokenizers,
    prompt: Union[str, List[str]],
    prompt_2: Union[str, List[str]],
    device: Optional[torch.device] = None,
    num_images_per_prompt: int = 1,
    prompt_embeds: Optional[torch.FloatTensor] = None,
    pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
    max_sequence_length: int = 512,
):

    device = device

    prompt = [prompt] if isinstance(prompt, str) else prompt
    if prompt is not None:
        batch_size = len(prompt)
    else:
        batch_size = prompt_embeds.shape[0]

    if prompt_embeds is None:
        prompt_2 = prompt_2 or prompt
        prompt_2 = [prompt_2] if isinstance(prompt_2, str) else prompt_2

        # We only use the pooled prompt output from the CLIPTextModel
        pooled_prompt_embeds = get_clip_prompt_embeds(
            text_encoders[0],
            tokenizers[0],
            prompt=prompt,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
        )
        prompt_embeds = get_t5_prompt_embeds(
            text_encoders[1],
            tokenizers[1],
            prompt=prompt_2,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
            device=device,
        )

    dtype = text_encoders[0].dtype

    return prompt_embeds, pooled_prompt_embeds


def get_train_dataset(args, accelerator):
    # Get the datasets: you can either provide your own training and evaluation files (see below)
    # or specify a Dataset from the hub (the dataset will be downloaded automatically from the datasets Hub).

    # In distributed training, the load_dataset function guarantees that only one local process can concurrently
    # download the dataset.
    if args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        dataset = load_dataset(
            args.dataset_name,
            args.dataset_config_name,
            cache_dir=args.cache_dir,
        )
    else:
        if args.train_data_json is not None:
            dataset = load_dataset(
                "json",
                data_files=args.train_data_json,
                cache_dir=args.cache_dir,
            )
        # See more about loading custom images at
        # https://huggingface.co/docs/datasets/v2.0.0/en/dataset_script

    # Preprocessing the datasets.
    # We need to tokenize inputs and targets.
    column_names = dataset["train"].column_names
    # 6. Get the column names for input/target.
    if args.image_column is None:
        image_column = column_names[0]
        logger.info(f"image column defaulting to {image_column}")
    else:
        image_column = args.image_column
        if image_column not in column_names:
            raise ValueError(
                f"`--image_column` value '{args.image_column}' not found in dataset columns. Dataset columns are: {', '.join(column_names)}"
            )

    if args.caption_column is None:
        caption_column = column_names[1]
        logger.info(f"caption column defaulting to {caption_column}")
    else:
        caption_column = args.caption_column
        if caption_column not in column_names:
            raise ValueError(
                f"`--caption_column` value '{args.caption_column}' not found in dataset columns. Dataset columns are: {', '.join(column_names)}"
            )

    with accelerator.main_process_first():
        train_dataset = dataset["train"].shuffle(seed=args.seed)
        if args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(args.max_train_samples))
    return train_dataset

# def prepare_train_dataset(dataset, accelerator):
#     image_transforms = transforms.Compose(
#         [
#             transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
#             transforms.CenterCrop(args.resolution),
#             transforms.ToTensor(),
#             transforms.Normalize([0.5], [0.5]),
#         ]
#     )
    
#     clip_image_processor = AutoProcessor.from_pretrained(args.siglip_path)

#     def preprocess_train(examples):
        
#         images = []
#         prompts = []
#         clip_images = []
#         drop_image_embeds = []
#         for i in range(len(examples[args.image_column])):
            
#             raw_image = Image.open(examples[args.image_column][i]).convert("RGB")
#             image = image_transforms(raw_image)

#             prompt = examples[args.caption_column][i]

#             clip_image = clip_image_processor(images=raw_image, return_tensors="pt").pixel_values

#             # drop
#             drop_image_embed = 0
#             rand_num = random.random()
#             if rand_num < 0.05:
#                 drop_image_embed = 1
                
#             images.append(image)
#             prompts.append(prompt)
#             clip_images.append(clip_image)
#             drop_image_embeds.append(drop_image_embed)
        
#         examples["pixel_values"] = images
#         examples["prompts"] = prompts
#         examples["clip_images"] = clip_images
#         examples["drop_image_embed"] = drop_image_embeds
        
#         return examples

#     with accelerator.main_process_first():
#         dataset = dataset.with_transform(preprocess_train)

#     return dataset





# 对 style_image 进行数据增强
def augment_style_image(style_image):
    # 定义数据增强的变换
    data_augmentations = transforms.Compose([
        transforms.RandomApply([
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)  # 随机调整亮度、对比度、饱和度和色调
        ], p=0.8),  # 80% 的概率应用颜色抖动
        transforms.RandomHorizontalFlip(p=0.5),  # 50% 的概率水平翻转
        transforms.RandomRotation(degrees=15),  # 随机旋转 ±15 度
        transforms.RandomResizedCrop(size=(style_image.size[1], style_image.size[0]), scale=(0.8, 1.0)),  # 保持图像原始大小比例
        transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5))  # 应用高斯模糊
    ])
    try:
        # 如果输入是 Tensor，则转换为 PIL 图像
        if isinstance(style_image, torch.Tensor):
            style_image_pil = to_pil_image(style_image.mul(255).byte())  # 恢复到 [0, 255] 的范围
            input_type = "tensor"
        else:
            style_image_pil = style_image  # 如果是 PIL，直接使用
            input_type = "pil"

        # 应用数据增强
        style_image_augmented = data_augmentations(style_image_pil)

        # 根据输入类型返回增强后的图像
        if input_type == "tensor":
            # 如果输入是 Tensor，则将增强后的图像转回 Tensor 格式，且保持像素范围在 [0, 255]
            style_image_augmented_tensor = to_tensor(style_image_augmented).mul(255).byte()
            return style_image_augmented_tensor
        else:
            # 如果输入是 PIL，则直接返回 PIL 格式
            return style_image_augmented

    except Exception as e:
        print(f"数据增强时出错: {e}")
        return style_image  # 如果出错，返回原始图像

        

def prepare_train_dataset(dataset, accelerator):
    image_transforms = transforms.Compose(
        [
            transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(args.resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )
    
    clip_image_processor = AutoProcessor.from_pretrained(args.siglip_path)

    def preprocess_train(examples):
        
        images = []
        clip_images = []
        prompts = []
        drop_image_embeds = []
        mask_list = []
        for i in range(len(examples[args.image_column])):
            try:
            # 尝试处理图像
                padded_image = crop_resize_and_pad(image_path=examples[args.image_column][j], bbox=examples['bbox'][i])
                # right_text = examples["right_text"][i]
                # 格式化输出
                style_image = None

                ground_truth = Image.open(examples[args.image_column][i])
                # ground_truth
                ground_truth = image_transforms(ground_truth)


                # ********************
                # Load the original image to get its dimensions
                _, image_width, image_height = ground_truth.size()

                # 创建一个空白的 mask (值全为 0，大小与图像一致)
                mask_array = np.zeros((image_height, image_width), dtype=np.uint8)
                bbox = examples['bbox'][i]
                x1, y1 = int(bbox[0][0]), int(bbox[0][1])  # 左上角
                x2, y2 = int(bbox[2][0]), int(bbox[2][1])  # 右下角
                mid_x = (x1 + x2) // 2 
                # 随机选择一种 mask 模式
                # mask_type = np.random.choice([1, 2, 3])  # 1: 右边, 2: 左边, 3: 全部
                # mask_type = np.random.choice([3])  # 1: 右边, 2: 左边, 3: 全部
                mask_type = 3
                if mask_type == 1:  # 右边
                    style_image = left_part_resized
                    mask_array[y1:y2, mid_x:x2] = 1
                    mask_tensor = torch.from_numpy(mask_array).unsqueeze(dim=0)
                    right_text = examples["right_text"][i]
                    prompt = f"The text is '{right_text.strip()}'"

                elif mask_type == 2:  # 左边
                    style_image = right_part_resized
                    mask_array[y1:y2, x1:mid_x] = 1
                    mask_tensor = torch.from_numpy(mask_array).unsqueeze(dim=0)
                    left_text = examples["left_text"][i]
                    prompt = f"The text is '{left_text.strip()}'"

                elif mask_type == 3:  # 全部
                    style_image = padded_image

                    mask_array[y1:y2, x1:x2] = 1
                    mask_tensor = torch.from_numpy(mask_array).unsqueeze(dim=0)
                    full_text = examples["full_text"][j]
                    prompt = f"The text is '{full_text.strip()}'"
                # 对 style_image 进行数据增强

                # style_image = augment_style_image(style_image)
                
                style_clip_image = clip_image_processor(images=style_image, return_tensors="pt").pixel_values
                # ********************

            except Exception as e:
                print(f"Skipping image {i} due to error: {e}")
                # 不断随机尝试新的图像，直到成功
                while True:
                    try:
                        j = random.randrange(0, len(examples[args.image_column]))
                        padded_image = crop_resize_and_pad(image_path=examples[args.image_column][j], bbox=examples['bbox'][i])
                        # right_text = examples["right_text"][i]
                        # 格式化输出
                        style_image = None

                        ground_truth = Image.open(examples[args.image_column][j])
                        # ground_truth
                        ground_truth = image_transforms(ground_truth)


                        # ********************
                        # Load the original image to get its dimensions
                        _, image_width, image_height = ground_truth.size()

                        # 创建一个空白的 mask (值全为 0，大小与图像一致)
                        mask_array = np.zeros((image_height, image_width), dtype=np.uint8)
                        bbox = examples['bbox'][j]
                        x1, y1 = int(bbox[0][0]), int(bbox[0][1])  # 左上角
                        x2, y2 = int(bbox[2][0]), int(bbox[2][1])  # 右下角
                        mid_x = (x1 + x2) // 2 
                        # 随机选择一种 mask 模式
                        # mask_type = np.random.choice([1, 2, 3])  # 1: 右边, 2: 左边, 3: 全部
                        # mask_type = np.random.choice([3])  # 1: 右边, 2: 左边, 3: 全部
                        mask_type = 3
                        if mask_type == 1:  # 右边
                            style_image = left_part_resized
                            mask_array[y1:y2, mid_x:x2] = 1
                            mask_tensor = torch.from_numpy(mask_array).unsqueeze(dim=0)
                            right_text = examples["right_text"][j]
                            prompt = f"The text is '{right_text.strip()}'"

                        elif mask_type == 2:  # 左边
                            style_image = right_part_resized
                            mask_array[y1:y2, x1:mid_x] = 1
                            mask_tensor = torch.from_numpy(mask_array).unsqueeze(dim=0)
                            left_text = examples["left_text"][j]
                            prompt = f"The text is '{left_text.strip()}'"

                        elif mask_type == 3:  # 全部

                            style_image = padded_image

                            mask_array[y1:y2, x1:x2] = 1
                            mask_tensor = torch.from_numpy(mask_array).unsqueeze(dim=0)
                            full_text = examples["full_text"][j]
                            prompt = f"The text is '{full_text.strip()}'"
                        
                        # 对 style_image 进行数据增强
                        # style_image = augment_style_image(style_image)
                   
                        style_clip_image = clip_image_processor(images=style_image, return_tensors="pt").pixel_values
                        # ********************


                        break  # 内层成功后退出
                    except Exception as eo:
                        print(f"Skipping image {j} again due to error: {eo}")
                        continue  # 内层继续尝试
            # drop
            drop_image_embed = 0
            rand_num = random.random()
            if rand_num < 0.05:
                drop_image_embed = 1
                
            images.append(ground_truth)
            prompts.append(prompt)
            clip_images.append(style_clip_image)
            drop_image_embeds.append(drop_image_embed)
            mask_list.append(mask_tensor)
        
        examples["pixel_values"] = images
        examples["prompts"] = prompts
        examples["clip_images"] = clip_images
        examples["drop_image_embed"] = drop_image_embeds
        examples['mask'] = mask_list
        
        return examples
    

    with accelerator.main_process_first():
        dataset = dataset.with_transform(preprocess_train)

    return dataset

def collate_fn(examples):
    
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

    prompts = [example["prompts"] for example in examples]
    
    clip_images = torch.cat([example["clip_images"] for example in examples])
    clip_images = clip_images.to(memory_format=torch.contiguous_format).float()
    
    drop_image_embeds = [example["drop_image_embed"] for example in examples]

    mask = torch.stack([example["mask"] for example in examples])
    mask = mask.to(memory_format=torch.contiguous_format).float()
    
    return {
        "pixel_values": pixel_values,
        "prompts": prompts,
        "clip_images": clip_images,
        "drop_image_embeds": drop_image_embeds,
        "mask": mask,
    }

def pack_latents(latents, batch_size, num_channels_latents, height, width):
    latents = latents.view(batch_size, num_channels_latents, height // 2, 2, width // 2, 2)
    latents = latents.permute(0, 2, 4, 1, 3, 5)
    latents = latents.reshape(batch_size, (height // 2) * (width // 2), num_channels_latents * 4)

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
    latent_image_ids = torch.zeros(height // 2, width // 2, 3)
    latent_image_ids[..., 1] = latent_image_ids[..., 1] + torch.arange(height // 2)[:, None]
    latent_image_ids[..., 2] = latent_image_ids[..., 2] + torch.arange(width // 2)[None, :]

    latent_image_id_height, latent_image_id_width, latent_image_id_channels = latent_image_ids.shape

    latent_image_ids = latent_image_ids[None, :]
    latent_image_ids = latent_image_ids.reshape(
        latent_image_id_height * latent_image_id_width, latent_image_id_channels
    )

    return latent_image_ids.to(device=device, dtype=dtype)

class ImageProjModel(nn.Module):
    """Projection Model
    https://github.com/tencent-ailab/IP-Adapter/blob/main/ip_adapter/ip_adapter.py#L28
    """

    def __init__(
        self, 
        joint_attention_dim=1024, 
        embeddings_dim=768, 
        num_tokens=4
    ):
        super().__init__()
        self.joint_attention_dim = joint_attention_dim
        self.num_tokens = num_tokens
        self.proj = nn.Linear(embeddings_dim, self.num_tokens * self.joint_attention_dim)
        self.norm = nn.LayerNorm(self.joint_attention_dim)

    def forward(self, image_embeds):
        embeds = image_embeds
        extra_context_tokens = self.proj(embeds).reshape(
            -1, self.num_tokens, self.joint_attention_dim
        )
        extra_context_tokens = self.norm(extra_context_tokens)
        return extra_context_tokens

class MLPProjModel(torch.nn.Module):
    def __init__(self, cross_attention_dim=768, id_embeddings_dim=512, num_tokens=4):
        super().__init__()
        
        self.cross_attention_dim = cross_attention_dim
        self.num_tokens = num_tokens
        
        self.proj = torch.nn.Sequential(
            torch.nn.Linear(id_embeddings_dim, id_embeddings_dim*2),
            torch.nn.GELU(),
            torch.nn.Linear(id_embeddings_dim*2, cross_attention_dim*num_tokens),
        )
        self.norm = torch.nn.LayerNorm(cross_attention_dim)
        
    def forward(self, id_embeds):
        x = self.proj(id_embeds)
        x = x.reshape(-1, self.num_tokens, self.cross_attention_dim)
        x = self.norm(x)
        return x
    
def main(args):
    
    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
    kwargs_1 = InitProcessGroupKwargs(timeout=timedelta(seconds=5400))
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[kwargs,kwargs_1],
    )
    
    allow_adam_bf16 = args.allow_adam_bf16

    # Disable AMP for MPS.
    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError("Make sure to install wandb if you want to use it for logging during training.")

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # Load the tokenizers
    tokenizer_one = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=args.revision,
    )
    tokenizer_two = T5TokenizerFast.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer_2",
        revision=args.revision,
    )
    # import correct text encoder classes
    text_encoder_cls_one = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision
    )
    text_encoder_cls_two = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision, subfolder="text_encoder_2"
    )
    # Load scheduler and models
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )
    noise_scheduler_copy = copy.deepcopy(noise_scheduler)
    text_encoder_one, text_encoder_two = load_text_encoders(
        text_encoder_cls_one, text_encoder_cls_two
    )
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        revision=args.revision,
        variant=args.variant,
    )
    transformer = FluxTransformer2DModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="transformer", revision=args.revision, variant=args.variant
    )
    vae.requires_grad_(False)
    transformer.requires_grad_(False)
    text_encoder_one.requires_grad_(False)
    text_encoder_two.requires_grad_(False)
    
    # For mixed precision training we cast all non-trainable weights (vae, non-lora text_encoder and non-lora transformer) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    
    #ip-adapter (newly added)
    image_encoder = SiglipVisionModel.from_pretrained(args.siglip_path)
    image_encoder.requires_grad_(False)
    
    # more image tokens
    num_tokens = 128
    image_proj_model = MLPProjModel(
        cross_attention_dim=transformer.config.joint_attention_dim, # 4096
        id_embeddings_dim=1152, 
        num_tokens=num_tokens,
    )
    
    ip_attn_procs = {} # 19+38=57
    for name, _ in transformer.attn_processors.items():
        #if name.startswith("single_transformer_blocks."):
        #if name.startswith("transformer_blocks."):
        if name.startswith("transformer_blocks.") or name.startswith("single_transformer_blocks"):
            ip_attn_procs[name] = IPAFluxAttnProcessor2_0(
                hidden_size=transformer.config.num_attention_heads * transformer.config.attention_head_dim,
                cross_attention_dim=transformer.config.joint_attention_dim,
                num_tokens=num_tokens,
            ).to(accelerator.device, dtype=weight_dtype)
        else:
            ip_attn_procs[name] = transformer.attn_processors[name]
    
    transformer.set_attn_processor(ip_attn_procs)
    
    if args.ip_adapter_path is not None:
        print(f"loading image_proj_model ...")
        state_dict = torch.load(args.ip_adapter_path, map_location="cpu")
        image_proj_model.load_state_dict(state_dict["image_proj"], strict=True)
        adapter_modules = torch.nn.ModuleList(transformer.attn_processors.values())
        adapter_modules.load_state_dict(state_dict["ip_adapter"])

    if torch.backends.mps.is_available() and weight_dtype == torch.bfloat16:
        # due to pytorch#99272, MPS does not yet support bfloat16.
        raise ValueError(
            "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
        )

    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder_one.to(accelerator.device, dtype=weight_dtype)
    text_encoder_two.to(accelerator.device, dtype=weight_dtype)
    image_encoder.to(accelerator.device, dtype=weight_dtype)
    
    transformer.to(accelerator.device, dtype=weight_dtype)
    image_proj_model.to(accelerator.device, dtype=weight_dtype)
    
    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()
        if args.train_text_encoder:
            text_encoder_one.gradient_checkpointing_enable()
            text_encoder_two.gradient_checkpointing_enable()

    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    def save_model_hook(models, weights, output_dir):
        state_dict = {}
        for model in models:
            if isinstance(model, ImageProjModel) or isinstance(model, MLPProjModel):
                state_dict['image_proj'] = unwrap_model(model).state_dict()
            elif isinstance(model, FluxTransformer2DModel):
                state_dict['ip_adapter'] = torch.nn.ModuleList(unwrap_model(model).attn_processors.values()).state_dict()
            else:
                model.save_pretrained(os.path.join(output_dir, sub_dir))

            # make sure to pop weight so that corresponding model is not saved again
            weights.pop()

        output_file = os.path.join(output_dir, 'ip-adapter.bin')
        torch.save(state_dict, output_file)
    
    def load_model_hook(models, input_dir):
        for _ in range(len(models)):
            # pop models so that they are not loaded again
            model = models.pop()

            # load diffusers style into model
            if isinstance(unwrap_model(model), FluxTransformer2DModel):
                load_model = FluxTransformer2DModel.from_pretrained(input_dir, subfolder="transformer")
                model.register_to_config(**load_model.config)

                model.load_state_dict(load_model.state_dict())
            elif isinstance(unwrap_model(model), (CLIPTextModel, CLIPTextModelWithProjection, T5EncoderModel)):
                try:
                    load_model = CLIPTextModel.from_pretrained(input_dir, subfolder="text_encoder")
                    model(**load_model.config)
                    model.load_state_dict(load_model.state_dict())
                except Exception:
                    try:
                        load_model = T5EncoderModel.from_pretrained(input_dir, subfolder="text_encoder_2")
                        model(**load_model.config)
                        model.load_state_dict(load_model.state_dict())
                    except Exception:
                        raise ValueError(f"Couldn't load the model of type: ({type(model)}).")
            else:
                raise ValueError(f"Unsupported model found: {type(model)=}")

            del load_model

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )
    
    # Optimization parameters
    # double: 562.115328 parameters (M)
    # single: 1040.270848 parameters (M)
    # double + sinlge: 1518.426368 parameters (M)
    params_to_optimize = [p for p in itertools.chain(transformer.parameters(), image_proj_model.parameters()) if p.requires_grad]
    print(sum([p.numel() for p in params_to_optimize]) / 1000000, 'parameters (M)')

    # Optimizer creation
    if not (args.optimizer.lower() == "prodigy" or args.optimizer.lower() == "adamw"):
        logger.warning(
            f"Unsupported choice of optimizer: {args.optimizer}.Supported optimizers include [adamW, prodigy]."
            "Defaulting to adamW"
        )
        args.optimizer = "adamw"

    if args.use_8bit_adam and not args.optimizer.lower() == "adamw":
        logger.warning(
            f"use_8bit_adam is ignored when optimizer is not set to 'AdamW'. Optimizer was "
            f"set to {args.optimizer.lower()}"
        )

    if args.optimizer.lower() == "adamw":
        if args.use_8bit_adam:
            try:
                import bitsandbytes as bnb
            except ImportError:
                raise ImportError(
                    "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
                )

            optimizer_class = bnb.optim.AdamW8bit
        else:
            if not allow_adam_bf16:
                optimizer_class = torch.optim.AdamW
            else:
                from adam_bfloat16 import AdamWBF16
                optimizer_class = AdamWBF16

        optimizer = optimizer_class(
            params_to_optimize,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )

    if args.optimizer.lower() == "prodigy":
        try:
            import prodigyopt
        except ImportError:
            raise ImportError("To use Prodigy, please install the prodigyopt library: `pip install prodigyopt`")

        optimizer_class = prodigyopt.Prodigy

        if args.learning_rate <= 0.1:
            logger.warning(
                "Learning rate is too low. When using prodigy, it's generally better to set learning rate around 1.0"
            )
        if args.train_text_encoder and args.text_encoder_lr:
            logger.warning(
                f"Learning rates were provided both for the transformer and the text encoder- e.g. text_encoder_lr:"
                f" {args.text_encoder_lr} and learning_rate: {args.learning_rate}. "
                f"When using prodigy only learning_rate is used as the initial learning rate."
            )
            # changes the learning rate of text_encoder_parameters_one and text_encoder_parameters_two to be
            # --learning_rate
            params_to_optimize[1]["lr"] = args.learning_rate
            params_to_optimize[2]["lr"] = args.learning_rate

        optimizer = optimizer_class(
            params_to_optimize,
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            beta3=args.prodigy_beta3,
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
            decouple=args.prodigy_decouple,
            use_bias_correction=args.prodigy_use_bias_correction,
            safeguard_warmup=args.prodigy_safeguard_warmup,
        )
    



    # Dataset and DataLoaders creation.
    train_dataset = get_train_dataset(args, accelerator)
    train_dataset = prepare_train_dataset(train_dataset, accelerator)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.dataloader_num_workers,
    )
    
    tokenizers = [tokenizer_one, tokenizer_two]
    text_encoders = [text_encoder_one, text_encoder_two]

    def compute_text_embeddings(prompt, prompt_2, text_encoders, tokenizers):
        with torch.no_grad():
            prompt_embeds, pooled_prompt_embeds = encode_prompt(
                text_encoders, tokenizers, 
                prompt=prompt, prompt_2=prompt_2,
                device=accelerator.device, max_sequence_length=args.max_sequence_length,
            )
            
            prompt_embeds = prompt_embeds.to(accelerator.device)
            pooled_prompt_embeds = pooled_prompt_embeds.to(accelerator.device)
            
        return prompt_embeds, pooled_prompt_embeds

    # Clear the memory here
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    # Prepare everything with our `accelerator`.
    transformer, image_proj_model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        transformer, image_proj_model, optimizer, train_dataloader, lr_scheduler
    )
    
    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_name = "flux-ip-adapter"
        accelerator.init_trackers(tracker_name, config=vars(args))

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0


    # Resume potentially from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
            # if not os.path.isdir(folder_path):
            #     path=None
        else:            
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            if len(dirs) != 0:
                dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
                
                # Automatically delete old checkpoints if there are more than 3
                if len(dirs) > 10:
                    oldest_checkpoint = dirs[0]  # The first one in the sorted list is the oldest
                    oldest_checkpoint_path = os.path.join(args.output_dir, oldest_checkpoint)
                    accelerator.print(f"Deleting old checkpoint: {oldest_checkpoint_path}")
                    try:
                        # Remove the directory and its contents
                        if os.path.isdir(oldest_checkpoint_path):
                            for root, _, files in os.walk(oldest_checkpoint_path, topdown=False):
                                for file in files:
                                    os.remove(os.path.join(root, file))
                                os.rmdir(root)
                            os.rmdir(oldest_checkpoint_path)
                        else:
                            os.remove(oldest_checkpoint_path)
                    except Exception as e:
                        accelerator.print(f"Failed to delete old checkpoint {oldest_checkpoint_path}: {e}")

                # Update the most recent checkpoint after deletion, if needed
                dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
                path = dirs[-2] if len(dirs) > 2 else None
            else: 
                path = None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            # import pdb;pdb.set_trace()
            # accelerator.print(f"Resuming from checkpoint {path}")
            # image_proj_state_dict = torch.load(os.path.join(args.output_dir, path, 'ip-adapter.bin'), map_location=accelerator.device)
            # image_proj_model.load_state_dict(image_proj_state_dict["image_proj"], strict=True)
            # ip_layers = torch.nn.ModuleList(transformer.attn_processors.values())
            # ip_layers.load_state_dict(image_proj_state_dict["ip_adapter"], strict=False)
            # global_step = int(path.split("-")[1])

            # resume_global_step = global_step * args.gradient_accumulation_steps
            # initial_global_step = global_step
            # first_epoch = global_step // num_update_steps_per_epoch
            # resume_step = resume_global_step % (
            #     num_update_steps_per_epoch * args.gradient_accumulation_steps
            # )
            # 打印恢复点信息
            accelerator.print(f"Resuming from checkpoint {path}")
            # 加载保存的 state_dict
            image_proj_state_dict = torch.load(
                os.path.join(args.output_dir, path, 'ip-adapter.bin'), 
                map_location=accelerator.device
            )

            # 判断是否为多卡模式
            is_multi_gpu = hasattr(image_proj_model, "module")

            # 调整 image_proj 的 state_dict key 以确保兼容性
            new_image_proj_state_dict = {}
            for key in image_proj_state_dict["image_proj"]:
                if key.startswith("module.") and not is_multi_gpu:
                    # 如果 key 带有 "module." 前缀，且当前不是多卡模式，去掉前缀
                    new_key = key[len("module."):]
                elif not key.startswith("module.") and is_multi_gpu:
                    # 如果 key 没有 "module." 前缀，但当前是多卡模式，添加前缀
                    new_key = f"module.{key}"
                else:
                    # 否则保持不变
                    new_key = key
                new_image_proj_state_dict[new_key] = image_proj_state_dict["image_proj"][key]

            # 加载调整后的 image_proj state_dict
            image_proj_model.load_state_dict(new_image_proj_state_dict, strict=True)

            # 调整 ip_adapter 的 state_dict key 以确保兼容性
            new_ip_adapter_state_dict = {}
            for key in image_proj_state_dict["ip_adapter"]:
                if key.startswith("module.") and not is_multi_gpu:
                    # 如果 key 带有 "module." 前缀，且当前不是多卡模式，去掉前缀
                    new_key = key[len("module."):]
                elif not key.startswith("module.") and is_multi_gpu:
                    # 如果 key 没有 "module." 前缀，但当前是多卡模式，添加前缀
                    new_key = f"module.{key}"
                else:
                    # 否则保持不变
                    new_key = key
                new_ip_adapter_state_dict[new_key] = image_proj_state_dict["ip_adapter"][key]

           # 加载调整后的 ip_adapter state_dict
            if isinstance(transformer, torch.nn.parallel.DistributedDataParallel):
                # 如果 transformer 是 DDP 模型，取出内部的原始模型
                ip_layers = torch.nn.ModuleList(transformer.module.attn_processors.values())
            else:
                # 否则直接使用 transformer
                ip_layers = torch.nn.ModuleList(transformer.attn_processors.values())

            # 加载 state_dict
            ip_layers.load_state_dict(new_ip_adapter_state_dict, strict=False)
            # 恢复训练的步数信息
            global_step = int(path.split("-")[1])
            resume_global_step = global_step * args.gradient_accumulation_steps
            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = resume_global_step % (
                num_update_steps_per_epoch * args.gradient_accumulation_steps
            )
    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    def get_sigmas(timesteps, n_dim=4, dtype=torch.float32):
        sigmas = noise_scheduler_copy.sigmas.to(device=accelerator.device, dtype=dtype)
        schedule_timesteps = noise_scheduler_copy.timesteps.to(accelerator.device)
        timesteps = timesteps.to(accelerator.device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma




    if args.init_ckpt is not None:
        # 加载预训练权重到 CPU，避免直接占用 GPU 显存
        image_proj_state_dict = torch.load(
            os.path.join(args.init_ckpt, 'ip-adapter.bin'),
            map_location="cpu"
        )

        # 判断是否为多卡模式
        is_multi_gpu = hasattr(image_proj_model, "module")

        # 动态调整 state_dict 的 key
        def adjust_state_dict_keys(state_dict, is_multi_gpu):
            """
            调整 `state_dict` 的 key，确保与模型的 key 匹配。
            - 如果是多卡模式，确保 key 含有 'module.' 前缀。
            - 如果是单卡模式，移除 'module.' 前缀。
            """
            adjusted_state_dict = {}
            for key in state_dict:
                if key.startswith("module.") and not is_multi_gpu:
                    # 如果 key 有 'module.' 前缀，但模型不是多卡模式，去掉 'module.'
                    new_key = key[len("module."):]
                elif not key.startswith("module.") and is_multi_gpu:
                    # 如果 key 没有 'module.' 前缀，但模型是多卡模式，加上 'module.'
                    new_key = f"{key}"
                else:
                    # 否则保持不变
                    new_key = key
                adjusted_state_dict[new_key] = state_dict[key]
            return adjusted_state_dict

        # 对 image_proj 和 ip_adapter 的 state_dict 进行调整
        new_image_proj_state_dict = adjust_state_dict_keys(image_proj_state_dict["image_proj"], is_multi_gpu)
        new_ip_adapter_state_dict = adjust_state_dict_keys(image_proj_state_dict["ip_adapter"], is_multi_gpu)

        # 加载调整后的 image_proj state_dict
        with torch.no_grad():  # 避免构建计算图，节省显存
            try:
                if is_multi_gpu:
                    image_proj_model.module.load_state_dict(new_image_proj_state_dict, strict=True)
                else:
                    image_proj_model.load_state_dict(new_image_proj_state_dict, strict=True)
            except RuntimeError as e:
                print(f"Error loading image_proj_model state_dict: {e}")
                print("Please check if the keys in the state_dict match the model's keys.")
                raise

        # 加载调整后的 ip_adapter state_dict
        if isinstance(transformer, torch.nn.parallel.DistributedDataParallel):
            # 如果 transformer 是 DDP 模型，取出内部的原始模型
            attn_processors = transformer.module.attn_processors
            attn_processors = torch.nn.ModuleList(transformer.module.attn_processors.values())
        else:
            # attn_processors = transformer.attn_processors
            attn_processors = torch.nn.ModuleList(transformer.attn_processors.values())

        # 将 attn_processors 的权重加载
        with torch.no_grad():
            try:
                missing_keys, unexpected_keys = attn_processors.load_state_dict(new_ip_adapter_state_dict, strict=False)
                if missing_keys:
                    print(f"Missing keys when loading ip_adapter state_dict: {missing_keys}")
                if unexpected_keys:
                    print(f"Unexpected keys when loading ip_adapter state_dict: {unexpected_keys}")
            except RuntimeError as e:
                print(f"Error loading ip_adapter state_dict: {e}")
                print("Please check if the keys in the state_dict match the model's keys.")
                raise



    for epoch in range(first_epoch, args.num_train_epochs):
        transformer.train()
        image_proj_model.train()

        for step, batch in enumerate(train_dataloader):
            models_to_accumulate = [transformer, image_proj_model]

            #resume
            if args.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                if step % args.gradient_accumulation_steps == 0:
                    progress_bar.update(1)
                continue
            with accelerator.accumulate(models_to_accumulate):
                
                pixel_values = batch["pixel_values"].to(dtype=vae.dtype)
                prompts = batch["prompts"]
                
                prompt_embeds, pooled_prompt_embeds = compute_text_embeddings(
                    prompts, prompts, text_encoders, tokenizers
                )
                
                bsz, channel, width, height = pixel_values.shape
                
                vae_scale_factor = 2 ** len(vae.config.block_out_channels)
                height = 2 * (int(height) // vae_scale_factor)
                width = 2 * (int(width) // vae_scale_factor)

                # Convert images to latent space
                model_input = vae.encode(pixel_values).latent_dist.sample()
                model_input = model_input * vae.config.scaling_factor
                model_input = model_input.to(dtype=weight_dtype) #  torch.Size([bsz, 16, 128, 128])
                                
                # Sample noise that we'll add to the latents
                noise = torch.randn_like(model_input)
                bsz = model_input.shape[0]

                # Sample a random timestep for each image
                # for weighting schemes where we sample timesteps non-uniformly
                u = compute_density_for_timestep_sampling(
                    weighting_scheme=args.weighting_scheme,
                    batch_size=bsz,
                    logit_mean=args.logit_mean,
                    logit_std=args.logit_std,
                    mode_scale=args.mode_scale,
                )
                indices = (u * noise_scheduler_copy.config.num_train_timesteps).long()
                timesteps = noise_scheduler_copy.timesteps[indices].to(device=model_input.device)

                # Add noise according to flow matching.
                sigmas = get_sigmas(timesteps, n_dim=model_input.ndim, dtype=model_input.dtype)
                noisy_model_input = sigmas * noise + (1.0 - sigmas) * model_input
                
                packed_noisy_latents = pack_latents(
                    noisy_model_input,
                    batch_size=model_input.shape[0],
                    num_channels_latents=model_input.shape[1],
                    height=model_input.shape[2],
                    width=model_input.shape[3],
                )
                
                guidance = torch.tensor([1.0], device=accelerator.device)
                guidance = guidance.expand(model_input.shape[0])

                
                img_ids = prepare_latent_image_ids(
                    model_input.shape[2],
                    model_input.shape[3],
                    accelerator.device,
                    weight_dtype,
                )
                
                text_ids = torch.zeros(
                    prompt_embeds.shape[1],
                    3,
                ).to(device=accelerator.device, dtype=weight_dtype)
                
                #****************************************************

                # mask_image = self.mask_processor.preprocess(mask_image, height=height, width=width)
                mask_image = batch["mask"].to(dtype=vae.dtype)
                
                masked_image = pixel_values.clone() * (1 - mask_image)
                masked_image = masked_image.to(device=accelerator.device, dtype=prompt_embeds.dtype)

                high, wid = pixel_values.shape[-2:]
                mask, masked_image_latents = prepare_mask_latents4training(
                    mask=mask_image,
                    masked_image=masked_image,
                    batch_size=model_input.shape[0],
                    num_channels_latents=vae.config.latent_channels,
                    num_images_per_prompt=1,
                    height=high,
                    width=wid,
                    dtype=prompt_embeds.dtype,
                    device=accelerator.device,
                    generator=None,
                    vae=vae,
                    vae_scale_factor=8,
                )
                masked_image_latents = torch.cat((masked_image_latents, mask), dim=-1)
                #****************************************************

                # IP-Adapter (newly added)
                with torch.no_grad():
                    image_embeds = image_encoder(batch["clip_images"].to(accelerator.device, dtype=weight_dtype)).pooler_output
                    image_encoder.config.output_hidden_states = True  
                    # image_embeds = image_encoder(batch["clip_images"].to(accelerator.device, dtype=weight_dtype)).hidden_states[-2]                  
                image_embeds_ = []
                for image_embed, drop_image_embed in zip(image_embeds, batch["drop_image_embeds"]):
                    if drop_image_embed == 1:
                        image_embeds_.append(torch.zeros_like(image_embed))
                    else:
                        image_embeds_.append(image_embed)
                image_embeds = torch.stack(image_embeds_)
                
                ip_tokens = image_proj_model(image_embeds) # torch.Size([bsz, num_tokens, 4096])
     
                # Predict the noise residual
                model_pred = transformer(
                    hidden_states=torch.cat((packed_noisy_latents, masked_image_latents), dim=2).to(
                        dtype=weight_dtype, device=accelerator.device
                    ),
                    timestep=timesteps.to(
                        dtype=weight_dtype, device=accelerator.device
                    )/1000,
                    guidance=guidance.to(
                        dtype=weight_dtype, device=accelerator.device
                    ),
                    pooled_projections=pooled_prompt_embeds.to(
                        device=accelerator.device, dtype=weight_dtype
                    ),
                    encoder_hidden_states=prompt_embeds.to(
                        device=accelerator.device, dtype=weight_dtype
                    ),
                    image_emb=ip_tokens.to(
                        device=accelerator.device, dtype=weight_dtype
                    ),
                    txt_ids=text_ids.to(
                        device=accelerator.device, dtype=weight_dtype
                    ),
                    img_ids=img_ids.to(
                        dtype=weight_dtype, device=accelerator.device
                    ),
                    joint_attention_kwargs=None,
                    return_dict=False,
                )[0]
                                                
                model_pred = unpack_latents(
                    model_pred,
                    height=model_input.shape[2] * 8,
                    width=model_input.shape[3] * 8,
                    vae_scale_factor=16,
                )
                
                # Follow: Section 5 of https://arxiv.org/abs/2206.00364.
                # Preconditioning of the model outputs.
                model_pred = model_pred * (-sigmas) + noisy_model_input
                # these weighting schemes use a uniform timestep sampling
                # and instead post-weight the loss
                weighting = compute_loss_weighting_for_sd3(weighting_scheme=args.weighting_scheme, sigmas=sigmas)

                # flow matching loss
                target = model_input

                # Compute regular loss.
                loss = torch.mean(
                    (weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1),
                    1,
                )
                loss = loss.mean()
                
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = [p for p in itertools.chain(transformer.parameters(), image_proj_model.parameters()) if p.requires_grad]
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                                
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                
            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    if global_step % args.checkpointing_steps == 0:
                    #if global_step % args.checkpointing_steps == 0 or global_step == 10:
                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

    accelerator.wait_for_everyone()
    accelerator.end_training()
    

if __name__ == "__main__":
    args = parse_args()
    main(args)