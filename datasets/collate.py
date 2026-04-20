"""Collate helpers."""

from __future__ import annotations

import torch


def stack_supervised_batch(batch: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    xs, ys, ws = zip(*batch, strict=True)
    return torch.stack(xs, dim=0), torch.stack(ys, dim=0), torch.stack(ws, dim=0)


def stack_ssl_batch(batch: list[tuple[torch.Tensor, torch.Tensor]]) -> tuple[torch.Tensor, torch.Tensor]:
    xs, ws = zip(*batch, strict=True)
    return torch.stack(xs, dim=0), torch.stack(ws, dim=0)
