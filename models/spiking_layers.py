"""1D spiking residual building blocks."""

from __future__ import annotations

import torch
import torch.nn as nn

from models.lif import LIFTime


class SpikingBasicBlock1d(nn.Module):
    """Conv-BN-LIF Conv-BN (+skip) LIF."""

    def __init__(
        self,
        channels: int,
        beta: float,
        threshold: float,
        surrogate_alpha: float,
        reset: str,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(channels)
        self.lif1 = LIFTime(beta=beta, threshold=threshold, surrogate_alpha=surrogate_alpha, reset=reset)

        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(channels)
        self.lif2 = LIFTime(beta=beta, threshold=threshold, surrogate_alpha=surrogate_alpha, reset=reset)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,C,T]
        identity = x
        z = self.bn1(self.conv1(x))
        z = self.lif1(z)
        z = self.bn2(self.conv2(z))
        z = z + identity
        z = self.lif2(z)
        return z


class SpikingStem1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        beta: float,
        threshold: float,
        surrogate_alpha: float,
        reset: str,
    ) -> None:
        super().__init__()
        pad = kernel_size // 2
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=pad, bias=False)
        self.bn = nn.BatchNorm1d(out_channels)
        self.lif = LIFTime(beta=beta, threshold=threshold, surrogate_alpha=surrogate_alpha, reset=reset)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.bn(self.conv(x))
        return self.lif(z)
