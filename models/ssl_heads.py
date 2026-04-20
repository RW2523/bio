"""AugPred binary heads on top of a pooled representation."""

from __future__ import annotations

import torch
import torch.nn as nn


class AugPredSSLHeads(nn.Module):
    """Three independent binary logits for Arrow-of-Time / Permutation / Time-warp."""

    def __init__(self, in_dim: int) -> None:
        super().__init__()
        self.head_aot = nn.Linear(in_dim, 1)
        self.head_perm = nn.Linear(in_dim, 1)
        self.head_tw = nn.Linear(in_dim, 1)

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.head_aot(z), self.head_perm(z), self.head_tw(z)
