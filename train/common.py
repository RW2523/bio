"""Shared training utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from utils.io import read_json
from utils.paths import project_root


def resolve_path(p: str | Path) -> Path:
    pp = Path(p)
    if pp.is_absolute():
        return pp
    return (project_root() / pp).resolve()


@dataclass(frozen=True)
class NormStats:
    mean: np.ndarray
    std: np.ndarray


def load_norm_stats(artifacts_dir: Path) -> NormStats:
    stats = read_json(artifacts_dir / "norm_stats.json")
    return NormStats(mean=np.asarray(stats["mean"], dtype=np.float32), std=np.asarray(stats["std"], dtype=np.float32))


def load_label_map(artifacts_dir: Path) -> dict[str, Any]:
    return read_json(artifacts_dir / "label_map.json")


def load_splits(artifacts_dir: Path) -> dict[str, list[int]]:
    return read_json(artifacts_dir / "splits.json")


def to_bct(x_btc: torch.Tensor) -> torch.Tensor:
    """[B,T,C] -> [B,C,T]"""
    if x_btc.ndim != 3:
        raise ValueError(f"Expected [B,T,C], got {tuple(x_btc.shape)}")
    return x_btc.transpose(1, 2).contiguous()


def freeze_module(m: nn.Module) -> None:
    m.eval()
    for p in m.parameters():
        p.requires_grad = False


def count_trainable_params(m: nn.Module) -> int:
    return int(sum(p.numel() for p in m.parameters() if p.requires_grad))


def log_module_trainable(logger, name: str, m: nn.Module) -> None:
    logger.info("%s trainable params: %d", name, count_trainable_params(m))


def make_loader(ds: Dataset, *, batch_size: int, shuffle: bool, num_workers: int, sampler=None) -> DataLoader:
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=False,
    )
