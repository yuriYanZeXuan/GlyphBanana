"""
GRPO (Group Relative Policy Optimization) Training Logic
This module implements the GRPO approach for diffusion models.
Reference: Original Calligrapher implementation
"""

import torch
import torch.nn.functional as F
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class GRPOTrainer:
    """
    GRPO trainer that implements group relative policy optimization.
    
    This is similar to PPO but adapted for diffusion models, using log probabilities
    and advantage values to compute policy gradient losses with clipping.
    """
    
    def __init__(
        self,
        clip_range: float = 0.2,
        kl_beta: float = 0.1,
        adv_clip_max: float = 5.0,
    ):
        """
        Args:
            clip_range: PPO clipping range for policy ratio
            kl_beta: Coefficient for KL divergence penalty
            adv_clip_max: Maximum value for advantage clipping
        """
        self.clip_range = clip_range
        self.kl_beta = kl_beta
        self.adv_clip_max = adv_clip_max
        
    def compute_grpo_loss(
        self,
        log_prob: torch.Tensor,
        old_log_prob: torch.Tensor,
        advantages: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute GRPO loss using log probabilities and advantages.
        
        Args:
            log_prob: Log probability from current policy
            old_log_prob: Log probability from old policy (from rollout)
            advantages: Advantage values for each sample
            
        Returns:
            Dictionary containing loss components and metrics
        """
        # Clip advantages
        advantages_clip = torch.clamp(
            advantages,
            -self.adv_clip_max,
            self.adv_clip_max,
        )
        
        # Compute ratio of probabilities
        ratio = torch.exp(log_prob - old_log_prob)
        
        # PPO-style clipped loss
        unclipped_loss = -advantages_clip * ratio
        clipped_loss = -advantages_clip * torch.clamp(
            ratio,
            1.0 - self.clip_range,
            1.0 + self.clip_range,
        )
        
        # Take the maximum (worst case) of clipped and unclipped
        policy_loss = torch.mean(torch.maximum(unclipped_loss, clipped_loss))
        
        # Prepare loss dict for logging
        loss_terms = {
            "policy_loss": policy_loss.detach(),
            "ratio_mean": ratio.mean().detach(),
            "ratio_std": ratio.std().detach(),
            "adv_mean": advantages_clip.mean().detach(),
            "adv_std": advantages_clip.std().detach(),
        }
        
        return policy_loss, loss_terms
    
    def compute_grpo_loss_with_kl(
        self,
        log_prob: torch.Tensor,
        old_log_prob: torch.Tensor,
        advantages: torch.Tensor,
        prev_sample_mean: torch.Tensor,
        prev_sample_mean_ref: torch.Tensor,
        std_dev_t: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute GRPO loss with KL penalty.
        
        Args:
            log_prob: Log probability from current policy
            old_log_prob: Log probability from old policy (from rollout)
            advantages: Advantage values for each sample
            prev_sample_mean: Mean of predicted next sample from current model
            prev_sample_mean_ref: Mean of predicted next sample from reference model
            std_dev_t: Standard deviation at timestep t
            
        Returns:
            Dictionary containing loss components and metrics
        """
        # Compute policy loss
        policy_loss, loss_terms = self.compute_grpo_loss(log_prob, old_log_prob, advantages)
        
        # Compute KL divergence (simplified as MSE in latent space)
        kl_loss = ((prev_sample_mean - prev_sample_mean_ref) ** 2).mean() / (2 * (std_dev_t ** 2).mean() + 1e-8)
        
        # Total loss
        total_loss = policy_loss + self.kl_beta * kl_loss
        
        # Add KL to loss terms
        loss_terms["kl_loss"] = kl_loss.detach()
        loss_terms["total_loss"] = total_loss.detach()
        
        return total_loss, loss_terms

