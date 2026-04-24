"""Learned attention pooling over the time axis for 1D conv features [B, C, T]."""

from __future__ import annotations

import torch
import torch.nn as nn


class AttentionTemporalPool1d(nn.Module):
    """Softmax weights over time; output shape ``[B, C]``."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.proj = nn.Conv1d(int(channels), int(channels), kernel_size=1, bias=True)

    def forward(self, x_bct: torch.Tensor) -> torch.Tensor:
        logits = self.proj(x_bct)
        w = torch.softmax(logits, dim=-1)
        return (x_bct * w).sum(dim=-1)
