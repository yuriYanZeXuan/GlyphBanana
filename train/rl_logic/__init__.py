"""
RL Training Logic Module

This module contains different RL training strategies for the Calligrapher project:
- GRPO: Group Relative Policy Optimization (original method)
- NFT: Negative Feedback Training (DiffusionNFT method)
"""

from .grpo_trainer import GRPOTrainer
from .nft_trainer import NFTTrainer, update_old_model

__all__ = [
    "GRPOTrainer",
    "NFTTrainer",
    "update_old_model",
]

