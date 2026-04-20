"""Linear classification head for frozen-backbone probing."""

from __future__ import annotations

import torch.nn as nn


class LinearProbeHead(nn.Module):
    def __init__(self, in_dim: int, num_classes: int) -> None:
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, z):  # noqa: ANN001
        return self.fc(z)
