"""Zero-Crossing Step-Forward (ZCSF)-style temporal encoder for low-frequency IMU streams."""

from __future__ import annotations

import torch
import torch.nn as nn


class ZCSFEncoder(nn.Module):
    """
    Encode each continuous channel into positive/negative event streams.

    For each axis, emit a positive event when the signal rises by at least ``step``
    beyond the running positive reference, and a negative event when it falls by at
    least ``step`` below the running negative reference.

    Input:  [B, C, T]
    Output: [B, 2*C, T]   (positive streams concatenated with negative streams)
    """

    def __init__(self, channels: int, *, step: float = 0.25, emit_start_event: bool = False) -> None:
        super().__init__()
        if step <= 0:
            raise ValueError("ZCSF `step` must be positive.")
        self.channels = int(channels)
        self.step = float(step)
        self.emit_start_event = bool(emit_start_event)
        self.out_channels = 2 * self.channels

    def forward(self, x_bct: torch.Tensor) -> torch.Tensor:
        if x_bct.ndim != 3:
            raise ValueError(f"Expected [B,C,T], got {tuple(x_bct.shape)}")
        b, c, t = x_bct.shape
        if c != self.channels:
            raise ValueError(f"Expected {self.channels} channels, got {c}")
        if t == 0:
            return torch.zeros((b, self.out_channels, 0), device=x_bct.device, dtype=x_bct.dtype)

        pos = torch.zeros((b, c, t), device=x_bct.device, dtype=x_bct.dtype)
        neg = torch.zeros((b, c, t), device=x_bct.device, dtype=x_bct.dtype)

        ref_pos = x_bct[:, :, 0].clone()
        ref_neg = x_bct[:, :, 0].clone()
        if self.emit_start_event:
            pos[:, :, 0] = (x_bct[:, :, 0] >= 0).to(x_bct.dtype)
            neg[:, :, 0] = (x_bct[:, :, 0] < 0).to(x_bct.dtype)

        for ti in range(1, t):
            xt = x_bct[:, :, ti]
            pos_fire = xt >= (ref_pos + self.step)
            neg_fire = xt <= (ref_neg - self.step)

            pos[:, :, ti] = pos_fire.to(x_bct.dtype)
            neg[:, :, ti] = neg_fire.to(x_bct.dtype)

            ref_pos = torch.where(pos_fire, xt, ref_pos)
            ref_neg = torch.where(neg_fire, xt, ref_neg)

        return torch.cat([pos, neg], dim=1)
