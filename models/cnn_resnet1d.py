"""Continuous Conv1D ResNet backbone for wearable time-series baselines."""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn


class CNNBasicBlock1d(nn.Module):
    """Conv-BN-ReLU residual block for continuous time-series features."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(channels)
        self.act1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(channels)
        self.act2 = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        z = self.act1(self.bn1(self.conv1(x)))
        z = self.bn2(self.conv2(z))
        return self.act2(z + identity)


class CNNStem1d(nn.Module):
    """Initial Conv-BN-ReLU projection from raw xyz channels."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int) -> None:
        super().__init__()
        pad = kernel_size // 2
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=pad, bias=False)
        self.bn = nn.BatchNorm1d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class CNNResNet1d(nn.Module):
    """
    Continuous ResNet-1D backbone expecting x shaped [B, C, T].

    The output contract intentionally matches SpikingResNet1d: temporal mean pooling returns
    a fixed embedding [B, D] so SSL heads and linear probes can be reused unchanged.
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 32,
        layers: Sequence[int] = (1, 1, 2),
        stem_kernel: int = 7,
    ) -> None:
        super().__init__()
        layers = tuple(int(x) for x in layers)

        self.stem = CNNStem1d(
            in_channels=in_channels,
            out_channels=base_channels,
            kernel_size=stem_kernel,
        )

        blocks: list[nn.Module] = []
        ch = int(base_channels)
        for stage_i, n in enumerate(layers):
            if stage_i > 0:
                new_ch = ch * 2
                blocks.append(
                    nn.Sequential(
                        nn.Conv1d(ch, new_ch, kernel_size=3, stride=2, padding=1, bias=False),
                        nn.BatchNorm1d(new_ch),
                        nn.ReLU(inplace=True),
                    )
                )
                ch = new_ch
            for _ in range(int(n)):
                blocks.append(CNNBasicBlock1d(channels=ch))

        self.blocks = nn.Sequential(*blocks)
        self.out_dim = ch

    def forward(self, x_bct: torch.Tensor) -> torch.Tensor:
        x = self.stem(x_bct)
        x = self.blocks(x)
        return torch.mean(x, dim=-1)
