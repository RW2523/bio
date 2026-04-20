"""Shared training utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import Optimizer

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


def load_splits(artifacts_dir: Path) -> dict[str, Any]:
    """Load `splits.json` (contains `train`/`val`/`test` lists and optional `split_seed`)."""
    return read_json(artifacts_dir / "splits.json")


def subject_split_ids(splits: dict[str, Any]) -> tuple[list[int], list[int], list[int]]:
    """Return train/val/test subject id lists; raises if keys missing or malformed."""
    try:
        train = [int(x) for x in splits["train"]]
        val = [int(x) for x in splits["val"]]
        test = [int(x) for x in splits["test"]]
    except KeyError as e:
        raise KeyError("splits.json must contain train, val, and test lists") from e
    return train, val, test


def assert_optimizer_excludes_module(optimizer: Optimizer, module: nn.Module) -> None:
    """Ensure no parameters from `module` are optimized (linear probe must not update backbone)."""
    excluded = {id(p) for p in module.parameters()}
    for group in optimizer.param_groups:
        for p in group["params"]:
            if id(p) in excluded:
                raise AssertionError("Optimizer must not include backbone parameters during linear probing.")


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
