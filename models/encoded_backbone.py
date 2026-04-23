"""Composable input-encoder + backbone wrapper."""

from __future__ import annotations

import torch
import torch.nn as nn


class EncodedBackbone(nn.Module):
    """Apply an input encoder before a downstream backbone, preserving `out_dim`."""

    def __init__(self, encoder: nn.Module, backbone: nn.Module) -> None:
        super().__init__()
        self.encoder = encoder
        self.backbone = backbone
        self.out_dim = int(backbone.out_dim)

    def forward(self, x_bct: torch.Tensor) -> torch.Tensor:
        x_enc = self.encoder(x_bct)
        return self.backbone(x_enc)
