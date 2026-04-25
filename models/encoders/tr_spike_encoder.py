"""Threshold-based Representation (TR) spike encoder for continuous time-series."""

from __future__ import annotations

import torch
import torch.nn as nn


class TRSpikeEncoder(nn.Module):
    """
    Encode continuous windows into spike trains from consecutive temporal differences.

    For each channel independently: given normalized ``x[..., t, c]``,
    ``delta[t] = x[..., t+1, c] - x[..., t, c]``.

    - **two_channel** (default): output has ``F = 2 * C`` features per timestep.
      Channels ``2*c`` and ``2*c+1`` are binary {0,1} for positive / negative threshold crossings.
    - **ternary**: output has ``F = C`` with values in ``{-1.0, 0.0, +1.0}``.

    Args:
        threshold: Absolute difference threshold when ``threshold_scale == "absolute"``.
        threshold_scale: ``"absolute"`` or ``"per_channel_std"``.
            For ``per_channel_std``, effective threshold per channel is
            ``threshold * std(delta_c)`` computed from the **current batch** (train-time noise;
            use modest ``threshold`` values). Alternatively pass ``delta_std`` buffer updated EMA
            outside this module.
        encoding: ``"two_channel"`` or ``"ternary"``.
        delta_std: Optional ``[C]`` tensor (or None) used when ``threshold_scale == "per_channel_std"``
            to multiply ``threshold`` instead of batch statistics.

    Input:
        ``x``: float tensor ``[B, T, C]`` (e.g. z-scored sensor window).

    Output:
        ``spikes``: float tensor ``[B, T-1, F]`` with values in ``{0,1}`` (two_channel) or ``{-1,0,1}`` (ternary).
    """

    def __init__(
        self,
        *,
        num_channels: int,
        threshold: float = 0.25,
        threshold_scale: str = "absolute",
        encoding: str = "two_channel",
        delta_std: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.num_channels = int(num_channels)
        self.threshold = float(threshold)
        self.threshold_scale = str(threshold_scale).strip().lower()
        self.encoding = str(encoding).strip().lower()
        if self.encoding not in {"two_channel", "ternary"}:
            raise ValueError(f"encoding must be 'two_channel' or 'ternary', got {encoding!r}")
        if self.threshold_scale not in {"absolute", "per_channel_std"}:
            raise ValueError(f"threshold_scale must be 'absolute' or 'per_channel_std', got {threshold_scale!r}")
        if delta_std is not None:
            self.register_buffer("delta_std", delta_std.float().clone().view(-1))
        else:
            self.register_buffer("delta_std", torch.empty(0))

    @property
    def out_features(self) -> int:
        return 2 * self.num_channels if self.encoding == "two_channel" else self.num_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected x [B,T,C], got shape {tuple(x.shape)}")
        b, t, c = x.shape
        if c != self.num_channels:
            raise ValueError(f"x has C={c} but encoder expects num_channels={self.num_channels}")
        if t < 2:
            raise ValueError("Need T>=2 for temporal difference encoding.")

        delta = x[:, 1:, :] - x[:, :-1, :]  # [B, T-1, C]

        if self.threshold_scale == "absolute":
            thr = torch.full_like(delta, self.threshold)
        else:
            if self.delta_std.numel() == c:
                std = self.delta_std.view(1, 1, c).clamp(min=1e-6)
                thr = self.threshold * std
            else:
                std = delta.float().std(dim=(0, 1), unbiased=False).view(1, 1, c).clamp(min=1e-6)
                thr = self.threshold * std

        if self.encoding == "ternary":
            out = torch.zeros_like(delta)
            out = torch.where(delta > thr, torch.ones_like(delta), out)
            out = torch.where(delta < -thr, -torch.ones_like(delta), out)
            return out

        pos = (delta > thr).to(delta.dtype)
        neg = (delta < -thr).to(delta.dtype)
        return torch.cat([pos, neg], dim=-1)
