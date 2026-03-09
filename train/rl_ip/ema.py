"""Exponential Moving Average for model parameters."""

import torch


class EMA:
    """EMA wrapper for model parameters."""

    def __init__(self, params, decay: float = 0.9999, interval: int = 1, device=None):
        self.params = [p.clone().detach().to(device) for p in params]
        self.decay = decay
        self.interval = interval
        self.device = device

    def _get_decay(self, step: int) -> float:
        """Adaptive decay: starts lower, increases to target."""
        return min((1 + step) / (10 + step), self.decay)

    @torch.no_grad()
    def step(self, params, step: int):
        """Update EMA parameters."""
        if (step + 1) % self.interval != 0:
            return

        one_minus = 1 - self._get_decay(step)

        for ema_p, p in zip(self.params, params):
            if p.requires_grad:
                if ema_p.device == p.device:
                    ema_p.add_(one_minus * (p - ema_p))
                else:
                    p_copy = p.detach().to(ema_p.device)
                    p_copy.sub_(ema_p)
                    p_copy.mul_(one_minus)
                    ema_p.add_(p_copy)

    def to(self, device=None, dtype=None):
        """Move EMA parameters to device/dtype."""
        self.device = device
        self.params = [
            p.to(device=device, dtype=dtype) if p.is_floating_point() else p.to(device=device)
            for p in self.params
        ]

    @torch.no_grad()
    def copy_to(self, params, store: bool = True):
        """Copy EMA parameters to model. Optionally store current params."""
        if store:
            self._stored = [p.detach().cpu() for p in params]

        for ema_p, p in zip(self.params, params):
            p.data.copy_(ema_p.to(p.device).data)

    @torch.no_grad()
    def restore(self, params):
        """Restore previously stored parameters."""
        for stored, p in zip(self._stored, params):
            p.data.copy_(stored.to(p.device))
        self._stored = None

    def state_dict(self):
        return {"decay": self.decay, "params": self.params}

    def load_state_dict(self, state: dict):
        self.decay = state.get("decay", self.decay)
        self.params = state.get("params", self.params)
        self.to(self.device)
