"""Simplified GRPO training for text-to-image generation."""

import argparse
import itertools
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed, gather_object
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.optimization import get_scheduler
from diffusers.training_utils import (
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
)
from PIL import Image
from tqdm.auto import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from .dataset import LTB_Dataset, collate_ltb
from .flux_ip.utils import (
    encode_prompt,
    get_sigmas,
    pack_latents,
    prepare_latent_image_ids,
    unpack_latents,
)
from .model import (
    load_text_encoders_and_tokenizers,
    load_vae_and_transformer,
    setup_ip_adapter,
)
from .rl_ip.ema import EMA
from .rl_ip.grpo_utils import sde_step
from .rl_ip.reward import RewardClient
from .rl_ip.stat_tracking import PerPromptStatTracker
from .rl_logic.grpo_trainer import GRPOTrainer
from .lora_utils import (
    apply_zimage_attention_lora_peft,
    iter_trainable_params,
    
    load_zimage_lora_peft,
    save_zimage_lora_peft,
)

logger = get_logger(__name__)


def encode_conditioning(
    *,
    args,
    transformer,
    enc1,
    enc2,
    tok1,
    tok2,
    prompts: List[str],
    device,
    dtype,
) -> Dict[str, object]:
    if args.model_type == "zimage":
        # Align with `train/zimage_ip/pipeline_z_image.py`:
        # - Use chat template
        # - Take `.hidden_states[-2]`
        # - Select valid tokens by attention_mask
        chat_prompts = []
        for p in prompts:
            messages = [{"role": "user", "content": p}]
            chat_prompts.append(
                tok1.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=True,
                )
            )

        text_inputs = tok1(
            chat_prompts,
            padding="max_length",
            max_length=args.max_sequence_length,
            truncation=True,
            return_tensors="pt",
        )
        input_ids = text_inputs.input_ids.to(device)
        prompt_masks = text_inputs.attention_mask.to(device).bool()

        out = enc1(
            input_ids=input_ids,
            attention_mask=prompt_masks,
            output_hidden_states=True,
            return_dict=True,
        )
        hs = out.hidden_states[-2]

        cap_dim = int(getattr(transformer.config, "cap_feat_dim", hs.shape[-1]))
        if hs.shape[-1] != cap_dim:
            raise ValueError(
                f"Z-Image cap_feat_dim={cap_dim}, but text_encoder hidden={hs.shape[-1]}. 请换匹配的 text_encoder。"
            )

        cap_feats = [hs[i][prompt_masks[i]].to(dtype=dtype) for i in range(hs.shape[0])]
        return {"cap_feats": cap_feats}

    # FLUX/Qwen-style: dual encoders (CLIP + T5) via encode_prompt
    if enc2 is not None and args.offload_text_encoder_two:
        enc2.to(device, dtype=dtype)
    embeds, pooled = encode_prompt([enc1, enc2], [tok1, tok2], prompts, device, args.max_sequence_length)
    if enc2 is not None and args.offload_text_encoder_two:
        enc2.to("cpu")
        torch.cuda.empty_cache()
    return {"embeds": embeds, "pooled": pooled}


def predict_noise(
    *,
    args,
    transformer,
    noisy: torch.Tensor,
    ts: torch.Tensor,
    cond: Dict[str, object],
    device,
    dtype,
    num_images_per_prompt: int = 1,
    guidance_scale: Optional[float] = None,
) -> torch.Tensor:
    b, c, h, w = noisy.shape

    if args.model_type == "zimage":
        cap_feats = cond["cap_feats"]
        assert isinstance(cap_feats, list)
        if num_images_per_prompt != 1:
            cap_feats = [f for f in cap_feats for _ in range(num_images_per_prompt)]

        x_list = [noisy[i].unsqueeze(1) for i in range(noisy.shape[0])]
        t = (ts / 1000).to(device=device, dtype=dtype)
        pred_list = transformer(x=x_list, t=t, cap_feats=cap_feats, return_dict=False)[0]
        return torch.stack([p[:, 0] for p in pred_list], dim=0)

    # FLUX/Qwen-style path
    packed = pack_latents(noisy, b, c, h, w)
    guidance = torch.ones(b, device=device) if guidance_scale is None else torch.full((b,), guidance_scale, device=device)
    img_ids = prepare_latent_image_ids(h, w, device, dtype)
    embeds = cond["embeds"]
    pooled = cond["pooled"]
    assert torch.is_tensor(embeds) and torch.is_tensor(pooled)
    text_ids = torch.zeros(embeds.shape[1], 3, device=device, dtype=dtype)
    pred = transformer(
        hidden_states=packed,
        timestep=(ts / 1000).to(device=device),
        guidance=guidance,
        pooled_projections=pooled,
        encoder_hidden_states=embeds,
        txt_ids=text_ids,
        img_ids=img_ids,
        return_dict=False,
    )[0]
    return unpack_latents(pred, h * 8, w * 8, 16)


@torch.no_grad()
def rollout(args, vae, transformer, encoders, tokenizers, scheduler, batch, dtype, device):
    """Generate trajectories and compute log probabilities."""
    n = args.rl_num_images_per_prompt
    prompts = batch["prompts"]

    enc1, enc2 = encoders
    tok1, tok2 = tokenizers
    cond = encode_conditioning(
        args=args,
        transformer=transformer,
        enc1=enc1,
        enc2=enc2,
        tok1=tok1,
        tok2=tok2,
        prompts=prompts,
        device=device,
        dtype=dtype,
    )

    # Init latents from noise (no image conditioning)
    shape = (len(prompts) * n, 16, args.resolution // 8, args.resolution // 8)
    latents = torch.randn(shape, device=device, dtype=dtype)

    scheduler.set_timesteps(args.rl_num_inference_steps, device=device, mu=args.rl_guidance_scale - 1.0)

    all_latents = [latents]
    all_log_probs = []
    pred_x0 = None

    for i, t in enumerate(tqdm(scheduler.timesteps, desc="Rollout", leave=False)):
        t_batch = t.repeat(latents.shape[0]).to(device)

        pred = predict_noise(
            args=args,
            transformer=transformer,
            noisy=latents,
            ts=t_batch,
            cond=cond,
            device=device,
            dtype=dtype,
            num_images_per_prompt=n,
            guidance_scale=args.rl_guidance_scale if args.model_type != "zimage" else None,
        )
        sigma = scheduler.sigmas[i].to(device).view(-1, 1, 1, 1)
        pred_x0 = pred * (-sigma) + latents

        latents, log_prob, _, _ = sde_step(scheduler, pred_x0, t_batch, latents)
        all_latents.append(latents)
        all_log_probs.append(log_prob)

    # Decode
    decode_from = pred_x0 if args.rl_reward_image_from == "pred_x0" else latents
    imgs = vae.decode((decode_from / vae.config.scaling_factor).to(vae.dtype)).sample
    imgs = torch.nan_to_num(imgs)
    imgs = (imgs / 2 + 0.5).clamp(0, 1)

    pil_imgs = [Image.fromarray((t.permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")) for t in imgs]

    return pil_imgs, all_latents, all_log_probs


def compute_logprob(args, vae, transformer, encoders, tokenizers, scheduler, sample, idx, dtype, device):
    """Compute log probability for a trajectory step."""
    prompts = sample["prompts"]
    enc1, enc2 = encoders
    tok1, tok2 = tokenizers
    cond = encode_conditioning(
        args=args,
        transformer=transformer,
        enc1=enc1,
        enc2=enc2,
        tok1=tok1,
        tok2=tok2,
        prompts=prompts,
        device=device,
        dtype=dtype,
    )

    latents = sample["latents"][:, idx]
    next_latents = sample["next_latents"][:, idx]
    t = scheduler.timesteps[idx]
    t_batch = t.repeat(latents.shape[0]).to(device)

    pred = predict_noise(
        args=args,
        transformer=transformer,
        noisy=latents,
        ts=t_batch,
        cond=cond,
        device=device,
        dtype=dtype,
        guidance_scale=args.rl_guidance_scale if args.model_type != "zimage" else None,
    )
    sigmas = get_sigmas(scheduler, t_batch, latents.ndim, latents.dtype, latents.device)
    pred_x0 = pred * (-sigmas) + latents

    _, log_prob, mean, std = sde_step(scheduler, pred_x0, t_batch, latents, next_latents)
    return log_prob, mean, std


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pretrained_model_name_or_path", required=True)
    p.add_argument("--output_dir", default="output")
    p.add_argument("--train_data_json", required=True)
    p.add_argument("--model_type", default="zimage", choices=["flux", "qwen", "zimage"])
    p.add_argument("--revision", type=str, default=None)
    p.add_argument("--variant", type=str, default=None)
    p.add_argument("--resolution", type=int, default=512)
    p.add_argument("--train_batch_size", type=int, default=1)
    p.add_argument("--max_train_steps", type=int, default=10000)
    p.add_argument("--max_train_samples", type=int, default=None)
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p.add_argument("--mixed_precision", default="fp16", choices=["no", "fp16", "bf16"])
    p.add_argument("--gradient_checkpointing", action="store_true")
    p.add_argument("--offload_text_encoder_two", action="store_true")
    p.add_argument("--use_8bit_adam", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_sequence_length", type=int, default=512)
    p.add_argument("--checkpointing_steps", type=int, default=500)
    p.add_argument("--resume_from_checkpoint", type=str, default=None)
    p.add_argument("--report_to", default="tensorboard")

    # Z-Image LoRA
    p.add_argument("--use_zimage_lora", action="store_true")
    p.add_argument("--lora_rank", type=int, default=8)
    p.add_argument("--lora_alpha", type=float, default=16.0)
    p.add_argument("--lora_dropout", type=float, default=0.0)
    p.add_argument(
        "--lora_path",
        type=str,
        default=None,
        help="Path to saved PEFT adapter dir (e.g. .../zimage_lora/). If omitted, a new LoRA adapter is created.",
    )

    # RL args
    p.add_argument("--use_rl", action="store_true")
    p.add_argument("--rl_warmup_steps", type=int, default=1000)
    p.add_argument("--rl_num_images_per_prompt", type=int, default=4)
    p.add_argument("--rl_num_batches_per_epoch", type=int, default=1)
    p.add_argument("--rl_num_inference_steps", type=int, default=20)
    p.add_argument("--rl_guidance_scale", type=float, default=3.5)
    p.add_argument("--rl_timestep_fraction", type=float, default=1.0)
    p.add_argument("--rl_num_inner_epochs", type=int, default=1)
    p.add_argument("--rl_grpo_clip_range", type=float, default=0.2)
    p.add_argument("--rl_kl_beta", type=float, default=0.1)
    p.add_argument("--rl_adv_clip_max", type=float, default=5.0)
    p.add_argument("--rl_reward_image_from", default="pred_x0", choices=["t0_latent", "pred_x0"])
    p.add_argument("--rl_per_prompt_stat_tracking", action="store_true")
    p.add_argument("--reward_server_url", default="http://127.0.0.1:8000/score")
    p.add_argument("--ocr_weight", type=float, default=0.7)
    p.add_argument("--vlm_weight", type=float, default=0.3)

    # EMA
    p.add_argument("--use_ema", action="store_true")
    p.add_argument("--ema_decay", type=float, default=0.9999)
    p.add_argument("--ema_update_interval", type=int, default=1)

    return p.parse_args()


def main():
    args = parse_args()

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_dir=Path(args.output_dir, "logs"),
    )

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    if args.seed is not None:
        set_seed(args.seed)

    # Load models
    tok1, tok2, enc1, enc2 = load_text_encoders_and_tokenizers(args)
    vae, transformer = load_vae_and_transformer(args)
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )

    vae.requires_grad_(False)
    transformer.requires_grad_(False)
    enc1.requires_grad_(False)
    if enc2 is not None:
        enc2.requires_grad_(False)

    dtype = torch.float16 if args.mixed_precision == "fp16" else torch.bfloat16
    vae.to(accelerator.device, dtype=dtype)
    enc1.to(accelerator.device, dtype=dtype)
    transformer.to(accelerator.device, dtype=dtype)

    if enc2 is not None:
        if args.offload_text_encoder_two:
            enc2.to("cpu")
        else:
            enc2.to(accelerator.device, dtype=dtype)

    if args.gradient_checkpointing:
        # For diffusers models (e.g. ZImageTransformer2DModel), setting the flag is not enough:
        # we must initialize the checkpointing function via `enable_gradient_checkpointing()`.
        if hasattr(transformer, "enable_gradient_checkpointing"):
            transformer.enable_gradient_checkpointing()
        else:
            transformer.gradient_checkpointing = True

        if hasattr(enc1, "gradient_checkpointing_enable"):
            enc1.gradient_checkpointing_enable()
        if enc2 is not None and hasattr(enc2, "gradient_checkpointing_enable"):
            enc2.gradient_checkpointing_enable()

    # Z-Image LoRA (inject AFTER .to so LoRA params match device/dtype)
    lora_active = bool(args.use_zimage_lora and args.model_type == "zimage")
    if args.use_zimage_lora and args.model_type != "zimage":
        raise ValueError("--use_zimage_lora is only supported when --model_type=zimage")

    # If resuming and no explicit LoRA path provided, load adapter from checkpoint dir.
    if lora_active and args.resume_from_checkpoint and not args.lora_path:
        ckpt_lora = Path(args.resume_from_checkpoint) / "zimage_lora"
        if ckpt_lora.exists():
            args.lora_path = str(ckpt_lora)
            logger.info(f"Auto-detected PEFT LoRA at {ckpt_lora} (from --resume_from_checkpoint).")

    if lora_active:
        if args.lora_path:
            transformer = load_zimage_lora_peft(transformer, Path(args.lora_path))
            logger.info(f"Loaded PEFT LoRA from {args.lora_path}.")
        else:
            transformer = apply_zimage_attention_lora_peft(
                transformer, r=args.lora_rank, alpha=args.lora_alpha, dropout=args.lora_dropout
            )
            logger.info("Injected PEFT LoRA into Z-Image Attention linear layers.")

    def trainable_params(m):
        if lora_active:
            return iter_trainable_params(m)
        return itertools.chain(*(p.parameters() for p in m.attn_processors.values()))

    # Optimizer
    if args.use_8bit_adam:
        import bitsandbytes as bnb
        opt_cls = bnb.optim.AdamW8bit
    else:
        opt_cls = torch.optim.AdamW

    params = trainable_params(transformer)
    optimizer = opt_cls(params, lr=args.learning_rate)

    # Dataset
    dataset = LTB_Dataset(args, accelerator)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.train_batch_size, shuffle=True,
        collate_fn=collate_ltb,
    )

    lr_scheduler = get_scheduler(
        "constant",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    # RL setup
    ema = None
    rl_trainer = None

    if args.use_rl:
        urls = args.reward_server_url.split(",")
        reward_client = RewardClient(urls, args.ocr_weight, args.vlm_weight)
        stat_tracker = PerPromptStatTracker() if args.rl_per_prompt_stat_tracking else None
        rl_trainer = GRPOTrainer(args.rl_grpo_clip_range, args.rl_kl_beta, args.rl_adv_clip_max)

        if args.use_ema:
            ema = EMA(list(trainable_params(transformer)), args.ema_decay, args.ema_update_interval, accelerator.device)

    transformer, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        transformer, optimizer, dataloader, lr_scheduler
    )

    def unwrap(m):
        return accelerator.unwrap_model(m)

    def save_hook(models, weights, output_dir):
        if not accelerator.is_main_process:
            return
        tfm = unwrap(transformer)
        if lora_active:
            outdir = Path(output_dir)
            save_zimage_lora_peft(outdir, tfm)
            # Save disk by skipping full transformer weights (base comes from pretrained).
            for i in range(len(models) - 1, -1, -1):
                if models[i] is transformer:
                    models.pop(i)
                    weights.pop(i)
                    break

    accelerator.register_save_state_pre_hook(save_hook)

    if args.resume_from_checkpoint:
        accelerator.load_state(Path(args.resume_from_checkpoint))

    total_batch = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    logger.info(f"Batch size: {total_batch}, Steps: {args.max_train_steps}")

    progress = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    global_step = 0
    train_iter = itertools.cycle(dataloader)

    while global_step < args.max_train_steps:
        # Supervised warmup
        if not args.use_rl or global_step < args.rl_warmup_steps:
            transformer.train()

            with accelerator.accumulate(transformer):
                batch = next(train_iter)

                cond = encode_conditioning(
                    args=args,
                    transformer=unwrap(transformer),
                    enc1=enc1,
                    enc2=enc2,
                    tok1=tok1,
                    tok2=tok2,
                    prompts=batch["prompts"],
                    device=accelerator.device,
                    dtype=dtype,
                )

                # VAE encode
                latents = vae.encode(batch["pixel_values"].to(dtype=vae.dtype)).latent_dist.sample() * vae.config.scaling_factor
                latents = latents.to(dtype=dtype)

                # Add noise
                noise = torch.randn_like(latents)
                u = compute_density_for_timestep_sampling("logit_normal", latents.shape[0], logit_mean=0.0, logit_std=1.0,mode_scale=1.29)
                ts = scheduler.timesteps[(u * scheduler.config.num_train_timesteps).long()].to(latents.device)
                sigmas = get_sigmas(scheduler, ts, latents.ndim, latents.dtype, latents.device)
                noisy = sigmas * noise + (1.0 - sigmas) * latents

                pred = predict_noise(
                    args=args,
                    transformer=transformer,
                    noisy=noisy,
                    ts=ts,
                    cond=cond,
                    device=accelerator.device,
                    dtype=dtype,
                    guidance_scale=None,  # warmup uses guidance=1 in FLUX; zimage ignores
                )
                pred = pred * (-sigmas) + noisy

                weight = compute_loss_weighting_for_sd3("logit_normal", sigmas)
                b = latents.shape[0]
                loss = torch.mean((weight * (pred - latents) ** 2).reshape(b, -1), 1).mean()

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(trainable_params(unwrap(transformer)), 1.0)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                global_step += 1
                progress.update(1)
                progress.set_postfix({"loss": loss.item(), "lr": lr_scheduler.get_last_lr()[0]})
                if accelerator.is_main_process:
                    accelerator.log({"loss": loss.item()}, step=global_step)

        # RL training
        else:
            logger.info("Starting RL rollout...")
            all_samples = []

            for _ in range(args.rl_num_batches_per_epoch):
                batch = next(train_iter)

                imgs, latents, log_probs = rollout(
                    args, vae, transformer, [enc1, enc2], [tok1, tok2],
                    scheduler, batch, dtype, accelerator.device
                )

                prompts = [p for p in batch["prompts"] for _ in range(args.rl_num_images_per_prompt)]
                rewards = reward_client.get_rewards_batch(images=imgs, prompts=prompts)

                latents_tensor = torch.stack(latents, dim=1)
                log_probs_tensor = torch.stack(log_probs, dim=1)
                rewards_tensor = torch.tensor([r["combined"] for r in rewards], device=accelerator.device)

                all_samples.append({
                    "prompts": prompts,
                    "latents": latents_tensor[:, :-1],
                    "next_latents": latents_tensor[:, 1:],
                    "log_probs": log_probs_tensor,
                    "rewards": rewards_tensor,
                })

            samples = {
                k: (torch.cat([s[k] for s in all_samples]) if torch.is_tensor(all_samples[0][k])
                    else [p for s in all_samples for p in s[k]])
                for k in all_samples[0].keys()
            }

            del all_samples
            torch.cuda.empty_cache()

            # Compute advantages
            gathered_rewards = accelerator.gather(samples["rewards"]).cpu().numpy()
            gathered_prompts = gather_object(samples["prompts"])

            if accelerator.is_main_process:
                if stat_tracker:
                    advantages = stat_tracker.update(gathered_prompts, gathered_rewards)
                else:
                    advantages = (gathered_rewards - gathered_rewards.mean()) / (gathered_rewards.std() + 1e-8)
                adv_tensor = torch.from_numpy(advantages).to(accelerator.device)
            else:
                adv_tensor = torch.empty(len(gathered_rewards), device=accelerator.device)

            accelerator.wait_for_everyone()
            torch.distributed.broadcast(adv_tensor, src=0)

            batch_size = len(samples["rewards"])
            start = accelerator.process_index * batch_size
            samples["advantages"] = adv_tensor[start:start + batch_size]

            # RL update
            logger.info("RL training...")
            num_train_ts = int(args.rl_num_inference_steps * args.rl_timestep_fraction)

            for _ in range(args.rl_num_inner_epochs):
                perm = torch.randperm(len(samples["prompts"]), device=accelerator.device)
                perm_list = perm.cpu().tolist()

                shuffled = {
                    k: (v[perm] if torch.is_tensor(v) else [v[i] for i in perm_list])
                    for k, v in samples.items()
                }

                for i in range(0, len(shuffled["prompts"]), args.train_batch_size):
                    mini = {k: (v[i:i + args.train_batch_size] if isinstance(v, (torch.Tensor, list)) else v)
                            for k, v in shuffled.items()}

                    for j in range(num_train_ts):
                        with accelerator.accumulate(transformer):
                            log_prob, _, _ = compute_logprob(
                                args, vae, transformer, [enc1, enc2], [tok1, tok2],
                                scheduler, mini, j, dtype, accelerator.device
                            )

                            loss, terms = rl_trainer.compute_grpo_loss(
                                log_prob=log_prob,
                                old_log_prob=mini["log_probs"][:, j],
                                advantages=mini["advantages"],
                            )

                            accelerator.backward(loss)
                            if accelerator.sync_gradients:
                                accelerator.clip_grad_norm_(trainable_params(unwrap(transformer)), 1.0)

                            optimizer.step()
                            lr_scheduler.step()
                            optimizer.zero_grad()

                            if args.use_ema and ema:
                                ema.step(list(trainable_params(unwrap(transformer))), global_step)

                        if accelerator.sync_gradients:
                            global_step += 1
                            progress.update(1)
                            progress.set_postfix({k: v.item() if torch.is_tensor(v) else v for k, v in terms.items()})
                            if accelerator.is_main_process:
                                accelerator.log({k: v.item() if torch.is_tensor(v) else v for k, v in terms.items()}, step=global_step)

                            if global_step % args.checkpointing_steps == 0:
                                path = Path(args.output_dir) / f"checkpoint-{global_step}"
                                accelerator.save_state(path)
                                logger.info(f"Saved checkpoint to {path}")

                            if global_step >= args.max_train_steps:
                                break
                    if global_step >= args.max_train_steps:
                        break
                if global_step >= args.max_train_steps:
                    break

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        accelerator.save_state(Path(args.output_dir) / "final_checkpoint")
        logger.info("Training complete")
    accelerator.end_training()


if __name__ == "__main__":
    main()
