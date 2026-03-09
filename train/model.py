import torch
import torch.nn as nn
from collections import OrderedDict
from contextlib import contextmanager
from typing import Dict, Iterable, Tuple
from transformers import (
    CLIPTokenizer,
    T5TokenizerFast,
    AutoTokenizer,
    AutoModelForCausalLM,
    CLIPTextModel,
    CLIPTextModelWithProjection,
    T5EncoderModel,
    SiglipVisionModel,
    PretrainedConfig,
)
from diffusers import AutoencoderKL
# Conditional imports based on model_type will be handled in functions
# from flux_ip.transformer_flux_inpainting import FluxTransformer2DModel
# from flux_ip.attention_processor import IPAFluxAttnProcessor2_0

def import_model_class_from_model_name_or_path(
    pretrained_model_name_or_path: str, revision: str, subfolder: str = "text_encoder"
):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path, subfolder=subfolder, revision=revision
    )
    model_class = text_encoder_config.architectures[0]
    if model_class == "CLIPTextModelWithProjection":
        return CLIPTextModelWithProjection
    elif model_class == "T5EncoderModel":
        return T5EncoderModel
    elif model_class == "CLIPTextModel":
        return CLIPTextModel
    elif model_class.endswith("ForCausalLM"):
        # e.g. Qwen3ForCausalLM (Z-Image caption encoder checkpoints)
        return AutoModelForCausalLM
    else:
        raise ValueError(f"{model_class} is not supported.")

def load_text_encoders_and_tokenizers(args):
    # Z-Image checkpoints typically only ship a single tokenizer/text_encoder.
    if args.model_type == "zimage":
        tokenizer_one = AutoTokenizer.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision, use_fast=True
        )
        text_encoder_cls_one = import_model_class_from_model_name_or_path(
            args.pretrained_model_name_or_path, args.revision, subfolder="text_encoder"
        )
        text_encoder_one = text_encoder_cls_one.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision, variant=args.variant
        )
        return tokenizer_one, None, text_encoder_one, None

    # FLUX/Qwen-style: dual encoders (CLIP + T5)
    tokenizer_one = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer", use_fast=True
    )
    tokenizer_two = T5TokenizerFast.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer_2", revision=args.revision
    )

    text_encoder_cls_one = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision
    )
    text_encoder_cls_two = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision, subfolder="text_encoder_2"
    )

    text_encoder_one = text_encoder_cls_one.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision, variant=args.variant
    )
    text_encoder_two = text_encoder_cls_two.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder_2", revision=args.revision, variant=args.variant
    )

    return tokenizer_one, tokenizer_two, text_encoder_one, text_encoder_two

def load_vae_and_transformer(args):
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision, variant=args.variant
    )
    
    if args.model_type == 'flux':
        # Import the custom transformer model that supports IP-Adapter (image_emb).
        # from diffusers.models import FluxTransformer2DModel as TransformerModel
        from .flux_ip.transformer_flux_inpainting import FluxTransformer2DModel as TransformerModel
    elif args.model_type == 'qwen':
        from .qwen_ip.transformer import QwenTransformer2DModel as TransformerModel
    elif args.model_type == 'zimage':
        from .zimage_ip.transformer_z_image import ZImageTransformer2DModel as TransformerModel
    else:
        raise ValueError(f"Unknown model_type: {args.model_type}")

    transformer = TransformerModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="transformer"
    )
    return vae, transformer

def load_image_encoder(args):
    image_encoder = SiglipVisionModel.from_pretrained(args.siglip_path)
    return image_encoder

class MLPProjModel(nn.Module):
    def __init__(self, cross_attention_dim=768, id_embeddings_dim=512, num_tokens=4):
        super().__init__()
        self.num_tokens = num_tokens
        self.cross_attention_dim = cross_attention_dim
        self.proj = nn.Sequential(
            nn.Linear(id_embeddings_dim, id_embeddings_dim * 2),
            nn.GELU(),
            nn.Linear(id_embeddings_dim * 2, cross_attention_dim * num_tokens),
        )
        self.norm = nn.LayerNorm(cross_attention_dim)
        
    def forward(self, id_embeds):
        x = self.proj(id_embeds)
        x = x.reshape(-1, self.num_tokens, self.cross_attention_dim)
        x = self.norm(x)
        return x


class AdapterizedMLPProjModel(nn.Module):
    def __init__(
        self,
        cross_attention_dim: int,
        id_embeddings_dim: int,
        num_tokens: int,
        adapter_names: Tuple[str, ...] = ("default", "old"),
        trainable_adapters: Tuple[str, ...] = ("default",),
    ):
        super().__init__()
        self._config = dict(
            cross_attention_dim=cross_attention_dim,
            id_embeddings_dim=id_embeddings_dim,
            num_tokens=num_tokens,
        )
        self.adapters = nn.ModuleDict()
        self.active_adapter = None
        for name in adapter_names:
            self.add_adapter(name, trainable=(name in trainable_adapters))
        if self.active_adapter is None:
            self.set_active_adapter(adapter_names[0])

    def add_adapter(self, name: str, trainable: bool = True, init_state: Dict[str, torch.Tensor] = None):
        module = MLPProjModel(**self._config)
        if init_state is not None:
            module.load_state_dict(init_state)
        for param in module.parameters():
            param.requires_grad_(trainable)
        self.adapters[name] = module
        if self.active_adapter is None:
            self.active_adapter = name

    def set_active_adapter(self, name: str):
        if name not in self.adapters:
            raise ValueError(f"Unknown adapter '{name}' for image projection model")
        self.active_adapter = name

    @contextmanager
    def use_adapter(self, name: str):
        previous = self.active_adapter
        self.set_active_adapter(name)
        try:
            yield
        finally:
            self.set_active_adapter(previous)

    def copy_adapter(self, source: str, target: str):
        if source not in self.adapters or target not in self.adapters:
            raise ValueError("Invalid adapter names for copy_adapter")
        self.adapters[target].load_state_dict(self.adapters[source].state_dict())

    def ema_update(self, source: str, target: str, decay: float):
        if source not in self.adapters or target not in self.adapters:
            raise ValueError("Invalid adapter names for ema_update")
        source_params = list(self.adapters[source].parameters())
        target_params = list(self.adapters[target].parameters())
        with torch.no_grad():
            for src, tgt in zip(source_params, target_params, strict=True):
                tgt.data.copy_(tgt.data * decay + src.data * (1.0 - decay))

    def get_adapter_parameters(self, name: str) -> Iterable[nn.Parameter]:
        if name not in self.adapters:
            raise ValueError(f"Unknown adapter '{name}' for image projection model")
        return self.adapters[name].parameters()

    def get_adapter_state_dict(self, name: str) -> Dict[str, torch.Tensor]:
        if name not in self.adapters:
            raise ValueError(f"Unknown adapter '{name}' for image projection model")
        return self.adapters[name].state_dict()

    def load_adapter_state_dict(self, name: str, state: Dict[str, torch.Tensor]):
        if name not in self.adapters:
            self.add_adapter(name, trainable=False)
        self.adapters[name].load_state_dict(state)

    def state_dict(self, destination=None, prefix: str = "", keep_vars: bool = False):
        return self.adapters[self.active_adapter].state_dict(destination, prefix, keep_vars)

    def load_state_dict(self, state_dict, strict: bool = True):
        if any(key.startswith("adapters.") for key in state_dict.keys()):
            return super().load_state_dict(state_dict, strict=strict)
        return self.adapters[self.active_adapter].load_state_dict(state_dict, strict=strict)

    def full_state_dict(self, destination=None, prefix: str = "", keep_vars: bool = False):
        return super().state_dict(destination, prefix, keep_vars)

    def forward(self, id_embeds):
        if self.active_adapter is None:
            raise RuntimeError("Active adapter is not set for the image projection model")
        return self.adapters[self.active_adapter](id_embeds)

class AdapterizedIPAttentionProcessor(nn.Module):
    supports_kwargs = True

    def __init__(
        self,
        hidden_size: int,
        cross_attention_dim: int,
        num_tokens: int,
        base_processor_cls,
        adapter_names: Tuple[str, ...] = ("default", "old"),
        trainable_adapters: Tuple[str, ...] = ("default",),
    ):
        super().__init__()
        self._config = dict(
            hidden_size=hidden_size,
            cross_attention_dim=cross_attention_dim,
            num_tokens=num_tokens,
        )
        self.base_processor_cls = base_processor_cls
        self.adapters = nn.ModuleDict()
        self.active_adapter = None
        self._trainable = set(trainable_adapters)
        self.supports_kwargs = True
        for name in adapter_names:
            self.add_adapter(name, trainable=(name in self._trainable))
        if self.active_adapter is None:
            self.set_active_adapter(adapter_names[0])

    def add_adapter(self, name: str, trainable: bool = True, init_state: Dict[str, torch.Tensor] = None):
        module = self.base_processor_cls(**self._config)
        if init_state is not None:
            module.load_state_dict(init_state)
        for param in module.parameters():
            param.requires_grad_(trainable)
        self.adapters[name] = module
        if self.active_adapter is None:
            self.active_adapter = name

    def set_active_adapter(self, name: str):
        if name not in self.adapters:
            raise ValueError(f"Unknown adapter '{name}' for attention processor")
        self.active_adapter = name

    def get_active_adapter(self) -> str:
        return self.active_adapter

    @contextmanager
    def use_adapter(self, name: str):
        previous = self.active_adapter
        self.set_active_adapter(name)
        try:
            yield
        finally:
            self.set_active_adapter(previous)

    def copy_adapter(self, source: str, target: str):
        if source not in self.adapters or target not in self.adapters:
            raise ValueError("Invalid adapter names for copy_adapter")
        self.adapters[target].load_state_dict(self.adapters[source].state_dict())

    def ema_update(self, source: str, target: str, decay: float):
        if source not in self.adapters or target not in self.adapters:
            raise ValueError("Invalid adapter names for ema_update")
        source_params = list(self.adapters[source].parameters())
        target_params = list(self.adapters[target].parameters())
        with torch.no_grad():
            for src, tgt in zip(source_params, target_params, strict=True):
                tgt.data.copy_(tgt.data * decay + src.data * (1.0 - decay))

    def get_adapter_parameters(self, name: str) -> Iterable[nn.Parameter]:
        if name not in self.adapters:
            raise ValueError(f"Unknown adapter '{name}' for attention processor")
        return self.adapters[name].parameters()

    def get_adapter_state_dict(self, name: str) -> Dict[str, torch.Tensor]:
        if name not in self.adapters:
            raise ValueError(f"Unknown adapter '{name}' for attention processor")
        return self.adapters[name].state_dict()

    def load_adapter_state_dict(self, name: str, state: Dict[str, torch.Tensor]):
        if name not in self.adapters:
            self.add_adapter(name, trainable=False)
        self.adapters[name].load_state_dict(state)

    def state_dict(self, destination=None, prefix: str = "", keep_vars: bool = False):
        return self.adapters[self.active_adapter].state_dict(destination, prefix, keep_vars)

    def load_state_dict(self, state_dict, strict: bool = True):
        if any(key.startswith("adapters.") for key in state_dict.keys()):
            return super().load_state_dict(state_dict, strict=strict)
        return self.adapters[self.active_adapter].load_state_dict(state_dict, strict=strict)

    def full_state_dict(self, destination=None, prefix: str = "", keep_vars: bool = False):
        return super().state_dict(destination, prefix, keep_vars)

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        image_emb=None,
        image_rotary_emb=None,
        **cross_attention_kwargs,
    ):
        return self.forward(
            attn,
            hidden_states,
            image_emb=image_emb,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
            image_rotary_emb=image_rotary_emb,
            **cross_attention_kwargs,
        )

    def forward(
        self,
        attn,
        hidden_states,
        image_emb=None,
        encoder_hidden_states=None,
        attention_mask=None,
        image_rotary_emb=None,
        **cross_attention_kwargs,
    ):
        if self.active_adapter is None:
            raise RuntimeError("Active adapter is not set for attention processor")

        if image_emb is None and "image_emb" in cross_attention_kwargs:
            image_emb = cross_attention_kwargs.pop("image_emb")
        if image_rotary_emb is None and "image_rotary_emb" in cross_attention_kwargs:
            image_rotary_emb = cross_attention_kwargs.pop("image_rotary_emb")

        processor = self.adapters[self.active_adapter]
        return processor(
            attn,
            hidden_states,
            image_emb,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
            image_rotary_emb=image_rotary_emb,
            **cross_attention_kwargs,
        )


def _default_adapter_names(args) -> Tuple[str, ...]:
    return ("default", "old")


def setup_ip_adapter(transformer, accelerator, weight_dtype, args):
    if args.model_type == 'flux':
        from .flux_ip.attention_processor import IPAFluxAttnProcessor2_0 as BaseAttentionProcessor
        num_tokens = 128
    elif args.model_type == 'qwen':
        from .qwen_ip.attention_processor import IPAQwenAttnProcessor as BaseAttentionProcessor
        # NOTE: Placeholder, adjust as needed for Qwen
        num_tokens = 16 
    else:
        raise ValueError(f"Unknown model_type: {args.model_type}")

    adapter_names = _default_adapter_names(args)
    image_proj_model = AdapterizedMLPProjModel(
        cross_attention_dim=transformer.config.joint_attention_dim,
        id_embeddings_dim=1152,
        num_tokens=num_tokens,
        adapter_names=adapter_names,
        trainable_adapters=(adapter_names[0],),
    )

    ip_attn_procs = {}
    # NOTE: This attention processor naming convention is based on Flux.
    # It might need to be adapted for the Qwen model's layer names.
    for name in transformer.attn_processors.keys():
        if name.startswith("transformer_blocks.") or name.startswith("single_transformer_blocks"):
            ip_attn_procs[name] = AdapterizedIPAttentionProcessor(
                hidden_size=transformer.config.num_attention_heads * transformer.config.attention_head_dim,
                cross_attention_dim=transformer.config.joint_attention_dim,
                num_tokens=num_tokens,
                base_processor_cls=BaseAttentionProcessor,
                adapter_names=adapter_names,
                trainable_adapters=(adapter_names[0],),
            )

    if ip_attn_procs:
        transformer.set_attn_processor(ip_attn_procs)

    # Ensure auxiliary adapters start with the same weights as default
    for processor in transformer.attn_processors.values():
        if hasattr(processor, "copy_adapter"):
            processor.copy_adapter(adapter_names[0], adapter_names[1])
    image_proj_model.copy_adapter(adapter_names[0], adapter_names[1])

    return image_proj_model


def set_ip_adapter_active(transformer, adapter_name: str):
    for processor in transformer.attn_processors.values():
        if hasattr(processor, "set_active_adapter"):
            processor.set_active_adapter(adapter_name)


@contextmanager
def use_ip_adapter(transformer, adapter_name: str):
    previous = []
    for processor in transformer.attn_processors.values():
        if hasattr(processor, "set_active_adapter"):
            old_adapter = processor.get_active_adapter() if hasattr(processor, "get_active_adapter") else None
            previous.append((processor, old_adapter))
            processor.set_active_adapter(adapter_name)
    try:
        yield
    finally:
        for processor, adapter in previous:
            if adapter is not None:
                processor.set_active_adapter(adapter)


def get_ip_adapter_parameter_pairs(transformer, source: str, target: str) -> Tuple[Iterable[nn.Parameter], Iterable[nn.Parameter]]:
    source_params = []
    target_params = []
    for processor in transformer.attn_processors.values():
        if hasattr(processor, "get_adapter_parameters"):
            source_params.extend(list(processor.get_adapter_parameters(source)))
            target_params.extend(list(processor.get_adapter_parameters(target)))
    return source_params, target_params


def get_ip_adapter_state_dict(transformer, adapter_name: str) -> Dict[str, Dict[str, torch.Tensor]]:
    state_dict = {}
    for name, processor in transformer.attn_processors.items():
        if hasattr(processor, "get_adapter_state_dict"):
            state_dict[name] = processor.get_adapter_state_dict(adapter_name)
    return state_dict


def load_ip_adapter_state_dict(transformer, adapter_name: str, state_dict: Dict[str, Dict[str, torch.Tensor]]):
    for name, processor in transformer.attn_processors.items():
        if hasattr(processor, "load_adapter_state_dict") and name in state_dict:
            processor.load_adapter_state_dict(adapter_name, state_dict[name])


def convert_legacy_ip_adapter_state_dict(state_dict: Dict[str, torch.Tensor], adapter_names: Tuple[str, ...] = ("default", "old")) -> Dict[str, torch.Tensor]:
    if not state_dict:
        return state_dict
    if any(".adapters." in key for key in state_dict.keys()):
        return state_dict

    converted = OrderedDict()
    for key, value in state_dict.items():
        if "." not in key:
            for adapter in adapter_names:
                converted[f"{key}.adapters.{adapter}"] = value.clone() if torch.is_tensor(value) else value
            continue
        module_idx, param = key.split(".", 1)
        for adapter in adapter_names:
            new_key = f"{module_idx}.adapters.{adapter}.{param}"
            converted[new_key] = value.clone() if torch.is_tensor(value) else value
    return converted
