from pathlib import Path
from typing import Iterable, Tuple

import torch.nn as nn

import peft  # type: ignore


def _find_zimage_attention_linear_names(transformer: nn.Module) -> Tuple[int, Tuple[str, ...]]:
    """
    Return (count, names) of attention linear submodules we want to LoRA-ize.
    Names are full module paths as returned by model.named_modules().
    """
    wanted_suffixes = (".to_q", ".to_k", ".to_v", ".to_out.0")
    names = []
    for name, module in transformer.named_modules():
        if isinstance(module, nn.Linear) and name.endswith(wanted_suffixes):
            names.append(name)
    return len(names), tuple(sorted(set(names)))


def apply_zimage_attention_lora_peft(
    transformer: nn.Module,
    r: int,
    alpha: float,
    dropout: float,
) -> nn.Module:
    """
    PEFT-based LoRA injection. Requires `peft` installed.

    Note: We compute exact target module names to robustly include `to_out.0`.
    """
    count, names = _find_zimage_attention_linear_names(transformer)
    if count == 0:
        raise RuntimeError("未找到可注入 LoRA 的 ZImage Attention Linear（to_q/to_k/to_v/to_out.0）")

    # target_modules in PEFT can be a list of module names; we pass exact names for safety.
    cfg = peft.LoraConfig(  # type: ignore[attr-defined]
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        target_modules=list(names),
    )
    return peft.get_peft_model(transformer, cfg)  # type: ignore[attr-defined]


def save_zimage_lora_peft(output_dir: Path, peft_model: nn.Module) -> Path:
    if not hasattr(peft_model, "save_pretrained"):
        raise TypeError("peft_model 不支持 save_pretrained()，请确认它是 PEFT 包装后的模型。")
    out = output_dir / "zimage_lora"
    out.mkdir(parents=True, exist_ok=True)
    peft_model.save_pretrained(out)  # type: ignore[call-arg]
    return out


def load_zimage_lora_peft(base_model: nn.Module, lora_dir: Path) -> nn.Module:
    return peft.PeftModel.from_pretrained(base_model, lora_dir, is_trainable=True)  # type: ignore[attr-defined]


def iter_trainable_params(model: nn.Module) -> Iterable[nn.Parameter]:
    # Works for both native and PEFT: only params with requires_grad=True
    return (p for p in model.parameters() if p.requires_grad)

