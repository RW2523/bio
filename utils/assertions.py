"""Sanity checks used across training/eval."""

from __future__ import annotations

from typing import Iterable

import torch
from torch import nn


def assert_disjoint_subject_sets(*sets: Iterable[int]) -> None:
    flat: list[int] = []
    for s in sets:
        flat.extend(list(s))
    if len(flat) != len(set(flat)):
        raise AssertionError("Subject leakage detected: splits are not disjoint.")


def assert_backbone_frozen(backbone: nn.Module) -> None:
    bad = [n for n, p in backbone.named_parameters() if p.requires_grad]
    if bad:
        raise AssertionError(f"Backbone not frozen; trainable params: {bad[:10]}...")


def assert_label_range(labels: torch.Tensor, num_classes: int) -> None:
    if labels.numel() == 0:
        return
    if int(labels.min()) < 0 or int(labels.max()) >= num_classes:
        raise AssertionError(f"Label out of range: min={int(labels.min())} max={int(labels.max())} C={num_classes}")


def assert_backbone_no_stored_gradients(backbone: nn.Module) -> None:
    """
    After `loss.backward()`, backbone parameters should not hold non-zero `.grad`
    when the backbone was excluded from the autograd path (frozen linear probe).
    """
    for name, p in backbone.named_parameters():
        if p.grad is None:
            continue
        if torch.isfinite(p.grad).all() and float(p.grad.detach().abs().sum()) > 0.0:
            raise AssertionError(f"Backbone parameter {name!r} received non-zero gradients during frozen probing.")
