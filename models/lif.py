"""Leaky integrate-and-fire (LIF) dynamics along the time axis of conv feature maps."""

from __future__ import annotations

import torch
import torch.nn as nn

from models.surrogate import spike_fn


class LIFTime(nn.Module):
    """
    Apply LIF spiking along dimension T for tensor shaped [B, H, T].

    This treats the conv feature map time index as the SNN time axis: each timestep receives
    an input current from the (already computed) conv output and updates membrane potential.
    """

    def __init__(
        self,
        beta: float = 0.9,
        threshold: float = 1.0,
        surrogate_alpha: float = 2.0,
        reset: str = "subtract",
    ) -> None:
        super().__init__()
        if reset not in {"subtract", "zero"}:
            raise ValueError("reset must be 'subtract' or 'zero'")
        self.beta = float(beta)
        self.threshold = float(threshold)
        self.surrogate_alpha = float(surrogate_alpha)
        self.reset = reset

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if z.ndim != 3:
            raise ValueError(f"Expected z [B,H,T], got {tuple(z.shape)}")
        b, h, t = z.shape
        mem = torch.zeros((b, h), device=z.device, dtype=z.dtype)
        spikes: list[torch.Tensor] = []
        for ti in range(t):
            mem = self.beta * mem + z[:, :, ti]
            sp = spike_fn(mem - self.threshold, self.surrogate_alpha)
            if self.reset == "subtract":
                mem = mem - sp * self.threshold
            else:
                mem = mem * (1.0 - sp)
            spikes.append(sp)
        return torch.stack(spikes, dim=2)
