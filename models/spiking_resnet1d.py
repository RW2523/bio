"""1D Spiking ResNet backbone for wearable time-series."""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn

from models.lif import LIFTime
from models.spiking_layers import SpikingBasicBlock1d, SpikingStem1d
from models.temporal_pooling import AttentionTemporalPool1d


class SpikingResNet1d(nn.Module):
    """
    Backbone expecting `x` shaped [B, C, T] (channel-first).

    Returns a pooled embedding of shape [B, D] using temporal mean pooling after the final stage.
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 32,
        layers: Sequence[int] = (1, 1, 2),
        stem_kernel: int = 7,
        beta: float = 0.9,
        threshold: float = 1.0,
        surrogate_alpha: float = 2.0,
        reset: str = "subtract",
        *,
        pooling: str = "mean",
        pool_bins: int = 1,
        debug_log_spike_rate: bool = False,
    ) -> None:
        super().__init__()
        layers = tuple(int(x) for x in layers)
        pooling = str(pooling).strip().lower()
        pool_bins = int(pool_bins)
        if pooling not in {"mean", "adaptive", "attention"}:
            raise ValueError(f"pooling must be 'mean', 'adaptive', or 'attention', got {pooling!r}")
        if pool_bins < 1:
            raise ValueError("pool_bins must be >= 1")
        self._pooling_mode = pooling
        self._pool_bins = pool_bins
        self.debug_log_spike_rate = bool(debug_log_spike_rate)
        self._last_spike_mean: float | None = None

        self.stem = SpikingStem1d(
            in_channels=in_channels,
            out_channels=base_channels,
            kernel_size=stem_kernel,
            beta=beta,
            threshold=threshold,
            surrogate_alpha=surrogate_alpha,
            reset=reset,
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
                        LIFTime(
                            beta=beta,
                            threshold=threshold,
                            surrogate_alpha=surrogate_alpha,
                            reset=reset,
                        ),
                    )
                )
                ch = new_ch
            for _ in range(int(n)):
                blocks.append(
                    SpikingBasicBlock1d(
                        channels=ch,
                        beta=beta,
                        threshold=threshold,
                        surrogate_alpha=surrogate_alpha,
                        reset=reset,
                    )
                )

        self.blocks = nn.Sequential(*blocks)
        self._temporal_pool: nn.AdaptiveAvgPool1d | None = None
        self._attn_pool: AttentionTemporalPool1d | None = None
        if pooling == "adaptive" and pool_bins > 1:
            self._temporal_pool = nn.AdaptiveAvgPool1d(pool_bins)
            self.out_dim = int(ch) * pool_bins
        elif pooling == "attention":
            self._attn_pool = AttentionTemporalPool1d(int(ch))
            self.out_dim = int(ch)
        else:
            self.out_dim = int(ch)

    def forward(self, x_bct: torch.Tensor) -> torch.Tensor:
        x = self.stem(x_bct)
        x = self.blocks(x)
        if self.debug_log_spike_rate and self.training:
            self._last_spike_mean = float(x.detach().mean().item())
        if self._attn_pool is not None:
            return self._attn_pool(x)
        if self._temporal_pool is not None:
            return self._temporal_pool(x).flatten(1)
        return torch.mean(x, dim=-1)
