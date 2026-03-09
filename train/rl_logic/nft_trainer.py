"""
DiffusionNFT Training Logic
This module implements the NFT (Negative Feedback Training) approach for diffusion models.
Reference: Edit-R1 implementation
"""

import torch
import torch.nn.functional as F
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class NFTTrainer:
    """
    DiffusionNFT trainer that implements negative feedback training.
    
    Key concepts:
    - Uses implicit negative prediction from a reference model
    - Applies adaptive weighting based on prediction quality
    - Includes KL divergence penalty to prevent drift from reference
    """
    
    def __init__(
        self,
        beta: float = 0.5,
        adv_clip_max: float = 5.0,
        kl_beta: float = 0.01,
        adv_mode: str = "normal",  # normal, positive_only, negative_only, one_only, binary
    ):
        """
        Args:
            beta: Mixing coefficient for positive/negative predictions
            adv_clip_max: Maximum value for advantage clipping
            kl_beta: Coefficient for KL divergence penalty
            adv_mode: How to process advantages (normal, positive_only, etc.)
        """
        self.beta = beta
        self.adv_clip_max = adv_clip_max
        self.kl_beta = kl_beta
        self.adv_mode = adv_mode
        
    def compute_nft_loss(
        self,
        forward_prediction: torch.Tensor,
        old_prediction: torch.Tensor,
        ref_prediction: torch.Tensor,
        x0: torch.Tensor,
        xt: torch.Tensor,
        t: torch.Tensor,
        advantages: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute NFT loss with implicit negative feedback.
        
        Args:
            forward_prediction: Prediction from current model
            old_prediction: Prediction from old/reference model (for implicit negative)
            ref_prediction: Prediction from frozen reference model (for KL penalty)
            x0: Clean target latents
            xt: Noisy latents at timestep t
            t: Timestep values
            advantages: Advantage values for each sample
            valid_mask: Optional mask for valid samples
            
        Returns:
            Dictionary containing loss components and metrics
        """
        loss_terms = {}
        
        # Prepare advantage values
        advantages_clip = torch.clamp(
            advantages,
            -self.adv_clip_max,
            self.adv_clip_max,
        )
        
        # Apply advantage mode
        if self.adv_mode == "positive_only":
            advantages_clip = torch.clamp(advantages_clip, 0, self.adv_clip_max)
        elif self.adv_mode == "negative_only":
            advantages_clip = torch.clamp(advantages_clip, -self.adv_clip_max, 0)
        elif self.adv_mode == "one_only":
            advantages_clip = torch.where(
                advantages_clip > 0,
                torch.ones_like(advantages_clip),
                torch.zeros_like(advantages_clip),
            )
        elif self.adv_mode == "binary":
            advantages_clip = torch.sign(advantages_clip)
        
        # Normalize advantages to [0, 1] range
        normalized_advantages = (advantages_clip / self.adv_clip_max) / 2.0 + 0.5
        r = torch.clamp(normalized_advantages, 0, 1)
        
        # Expand timestep for broadcasting
        t_expanded = t.view(-1, *([1] * (len(x0.shape) - 1)))
        
        # Compute positive prediction (weighted combination of forward and old)
        positive_prediction = (
            self.beta * forward_prediction
            + (1 - self.beta) * old_prediction.detach()
        )
        
        # Compute implicit negative prediction
        implicit_negative_prediction = (
            (1.0 + self.beta) * old_prediction.detach()
            - self.beta * forward_prediction
        )
        
        # Reconstruct x0 from predictions
        x0_prediction = xt - t_expanded * positive_prediction
        negative_x0_prediction = xt - t_expanded * implicit_negative_prediction
        
        # Compute adaptive weights
        with torch.no_grad():
            weight_factor = (
                torch.abs(x0_prediction.double() - x0.double())
                .mean(dim=tuple(range(1, x0.ndim)), keepdim=True)
                .clip(min=0.00001)
            )
            negative_weight_factor = (
                torch.abs(negative_x0_prediction.double() - x0.double())
                .mean(dim=tuple(range(1, x0.ndim)), keepdim=True)
                .clip(min=0.00001)
            )
        
        # Compute positive and negative losses
        positive_loss = ((x0_prediction - x0) ** 2 / weight_factor).mean(
            dim=tuple(range(1, x0.ndim))
        )
        negative_loss = ((negative_x0_prediction - x0) ** 2 / negative_weight_factor).mean(
            dim=tuple(range(1, x0.ndim))
        )
        
        # Apply valid mask if provided
        if valid_mask is not None:
            valid_mask = valid_mask.float()
            positive_loss = positive_loss * valid_mask
            negative_loss = negative_loss * valid_mask
        
        # Combine positive and negative losses with advantage weighting
        ori_policy_loss = (
            r * positive_loss / self.beta
            + (1.0 - r) * negative_loss / self.beta
        )
        
        # Mean by mask helper
        def mean_by_mask(x, mask):
            if mask is None:
                return x.mean()
            if mask.sum() == 0:
                return x.sum() * 0
            return x.sum() / mask.sum()
        
        policy_loss = mean_by_mask(ori_policy_loss * self.adv_clip_max, valid_mask)
        
        # KL divergence loss (between current and reference model)
        kl_div_loss = (
            (forward_prediction - ref_prediction) ** 2
        ).mean(dim=tuple(range(1, x0.ndim)))
        
        if valid_mask is not None:
            kl_div_loss = kl_div_loss * valid_mask
        
        kl_div_loss_mean = mean_by_mask(kl_div_loss, valid_mask)
        
        # Total loss
        total_loss = policy_loss + self.kl_beta * kl_div_loss_mean
        
        # Store loss components for logging
        loss_terms["policy_loss"] = policy_loss.detach()
        loss_terms["unweighted_policy_loss"] = mean_by_mask(ori_policy_loss, valid_mask).detach()
        loss_terms["kl_div_loss"] = kl_div_loss_mean.detach()
        loss_terms["kl_div"] = mean_by_mask(
            ((forward_prediction - ref_prediction) ** 2).mean(dim=tuple(range(1, x0.ndim))),
            valid_mask
        ).detach()
        loss_terms["old_kl_div"] = mean_by_mask(
            ((old_prediction - ref_prediction) ** 2).mean(dim=tuple(range(1, x0.ndim))),
            valid_mask
        ).detach()
        loss_terms["total_loss"] = total_loss.detach()
        loss_terms["x0_norm"] = torch.mean(x0 ** 2).detach()
        loss_terms["x0_norm_max"] = torch.max(x0 ** 2).detach()
        loss_terms["old_deviate"] = torch.mean((forward_prediction - old_prediction) ** 2).detach()
        loss_terms["old_deviate_max"] = torch.max((forward_prediction - old_prediction) ** 2).detach()
        
        return total_loss, loss_terms


def update_old_model(
    current_params,
    old_params,
    decay: float,
    global_step: int,
    decay_type: int = 0,
):
    """
    Update the old/reference model parameters with decay.
    
    Args:
        current_params: Current model parameters
        old_params: Old/reference model parameters to update
        decay: Base decay rate
        global_step: Current training step
        decay_type: Type of decay schedule (0, 1, or 2)
    """
    # Compute decay based on schedule
    if decay_type == 0:
        # No decay (keep old model fixed)
        actual_decay = 0.0
    elif decay_type == 1:
        # Linear warmup to small decay
        flat = 0
        uprate = 0.001
        uphold = 0.5
        if global_step < flat:
            actual_decay = 0.0
        else:
            actual_decay = min((global_step - flat) * uprate, uphold)
    elif decay_type == 2:
        # Linear warmup to high decay
        flat = 75
        uprate = 0.0075
        uphold = 0.999
        if global_step < flat:
            actual_decay = 0.0
        else:
            actual_decay = min((global_step - flat) * uprate, uphold)
    else:
        raise ValueError(f"Unknown decay_type: {decay_type}")
    
    # Update old parameters
    with torch.no_grad():
        for current_param, old_param in zip(current_params, old_params, strict=True):
            old_param.data.copy_(
                old_param.data * actual_decay + current_param.data * (1.0 - actual_decay)
            )

