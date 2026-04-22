"""Linear classification head for frozen-backbone probing."""

from __future__ import annotations

import torch
import torch.nn as nn


class LinearProbeHead(nn.Module):
    """
    Single linear layer on backbone embeddings.

    Optional ``embedding_batchnorm`` adds a trainable BatchNorm on the embedding
    before the linear readout (still only the head is trained; helps random/frozen
    backbones where feature scale varies across channels).
    """

    def __init__(self, in_dim: int, num_classes: int, *, embedding_batchnorm: bool = False) -> None:
        super().__init__()
        self.embedding_batchnorm = bool(embedding_batchnorm)
        self.bn = nn.BatchNorm1d(in_dim, affine=True) if self.embedding_batchnorm else None
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if self.bn is not None:
            z = self.bn(z)
        return self.fc(z)
