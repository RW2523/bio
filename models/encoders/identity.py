"""Identity input encoder for continuous sensor inputs."""

from __future__ import annotations

import torch
import torch.nn as nn


class IdentityEncoder(nn.Module):
    """Pass-through encoder for continuous [B, C, T] inputs."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.channels = int(channels)
        self.out_channels = self.channels

    def forward(self, x_bct: torch.Tensor) -> torch.Tensor:
        if x_bct.ndim != 3:
            raise ValueError(f"Expected [B,C,T], got {tuple(x_bct.shape)}")
        return x_bct
