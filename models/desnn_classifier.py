"""Supervised heads for reservoir features: deSNN-style prototypes vs linear probe."""

from __future__ import annotations

import torch
import torch.nn as nn


class DeSNNClassifier(nn.Module):
    """
    Learned class prototypes in feature space with negative squared-Euclidean logits.

    A practical simplification of rank-order / deSNN readouts: each class has a prototype ``p_c``;
    logits ``logit_c = -||z - p_c||^2`` (optionally scaled). ``z`` may be a short MLP of reservoir rates.
    """

    def __init__(
        self,
        in_dim: int,
        num_classes: int,
        *,
        hidden: int | None = 64,
        temperature: float = 1.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.temperature = float(temperature)
        p = float(dropout)
        if hidden is not None and int(hidden) > 0:
            h = int(hidden)
            layers: list[nn.Module] = [nn.Linear(in_dim, h), nn.ReLU(inplace=True)]
            if p > 0.0:
                layers.append(nn.Dropout(p))
            layers.extend([nn.Linear(h, h), nn.ReLU(inplace=True)])
            if p > 0.0:
                layers.append(nn.Dropout(p))
            self.encoder = nn.Sequential(*layers)
            emb_dim = h
        else:
            self.encoder = nn.Identity()
            emb_dim = in_dim
        self.prototypes = nn.Parameter(torch.randn(num_classes, emb_dim) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        z = torch.nn.functional.normalize(z, dim=-1, eps=1e-6)
        p = torch.nn.functional.normalize(self.prototypes, dim=-1, eps=1e-6)
        dist2 = torch.sum((z.unsqueeze(1) - p.unsqueeze(0)) ** 2, dim=-1)
        return -(dist2 / max(self.temperature, 1e-6))


def build_output_head(
    mode: str,
    *,
    reservoir_dim: int,
    num_classes: int,
    desnn_hidden: int | None = 64,
    desnn_temperature: float = 1.0,
    desnn_dropout: float = 0.0,
    probe_embedding_batchnorm: bool = False,
) -> nn.Module:
    """
    Factory for the supervised output module.

    Modes:
        - ``desnn``: prototype classifier on (optional) small MLP embedding of rates.
        - ``linear_probe_on_reservoir_features``: same interface as linear probes elsewhere.
    """
    m = str(mode).strip().lower()
    if m == "desnn":
        return DeSNNClassifier(
            reservoir_dim,
            num_classes,
            hidden=desnn_hidden,
            temperature=desnn_temperature,
            dropout=float(desnn_dropout),
        )
    if m in {"linear", "linear_probe", "linear_probe_on_reservoir_features", "linear_probe_on_reservoir"}:
        from models.linear_probe import LinearProbeHead

        return LinearProbeHead(
            reservoir_dim,
            num_classes=num_classes,
            embedding_batchnorm=bool(probe_embedding_batchnorm),
        )
    raise ValueError(f"Unknown classifier mode {mode!r}; use 'desnn' or 'linear_probe_on_reservoir_features'.")
