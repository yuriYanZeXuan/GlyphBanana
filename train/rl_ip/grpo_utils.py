"""SDE step with log probability computation for GRPO."""

import torch
from diffusers.utils.torch_utils import randn_tensor


def sde_step(scheduler, output, t, sample, prev=None, eps=1e-8):
    """Perform one SDE step and compute log probability.

    Args:
        scheduler: FlowMatchEulerDiscreteScheduler
        output: Model prediction (predicted x0)
        t: Current timestep
        sample: Current sample (x_t)
        prev: Previous sample (optional, computed if None)
        eps: Small constant for numerical stability

    Returns:
        (prev_sample, log_prob, mean, std)
    """
    device = sample.device

    # Get timestep indices on CPU
    t_cpu = t.cpu()
    unique_t, inv_idx = torch.unique(t_cpu, return_inverse=True)
    t_to_idx = {t.item(): i for i, t in enumerate(scheduler.timesteps)}
    idx = torch.tensor([t_to_idx[t.item()] for t in unique_t])

    # Get sigma values
    sigma = scheduler.sigmas[idx][inv_idx].to(device)
    sigma_next = scheduler.sigmas[(idx + 1).clamp(max=len(scheduler.sigmas) - 1)][inv_idx].to(device)

    # Last step: sigma_next = 0
    is_last = (idx[inv_idx] == len(scheduler.timesteps) - 1)
    sigma_next[is_last] = 0.0

    # Compute derivative and next sample
    derivative = (sample - output) / sigma.view(-1, 1, 1, 1)
    dt = (sigma_next - sigma).view(-1, 1, 1, 1)
    mean = sample + derivative * dt

    # For ODE, std is 0 (deterministic), use eps for numerical stability
    std = torch.sqrt(torch.zeros_like(sigma) * dt.squeeze())

    if prev is None:
        prev = mean

    # Log probability (without normalization constant)
    log_prob = -0.5 * torch.sum(
        ((prev.float() - mean.float()) / (std.view(-1, 1, 1, 1) + eps)) ** 2,
        dim=[1, 2, 3],
    )

    return prev, log_prob, mean, std
