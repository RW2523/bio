"""Shared training utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

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


def resolve_compute_device(name: str) -> torch.device:
    """
    Resolve a config/CLI device string to `torch.device`.

    - ``cuda`` / ``gpu`` require ``torch.cuda.is_available()``.
    - Other strings (e.g. ``cpu``, ``cuda:1``) are passed to ``torch.device``.
    """
    key = (name or "cuda").strip().lower()
    if key in {"cuda", "gpu"}:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "compute_device is set to CUDA but torch.cuda.is_available() is False. "
                "Install a CUDA-enabled PyTorch build with a visible NVIDIA GPU, or set compute_device to 'cpu' in YAML."
            )
        return torch.device("cuda")
    return torch.device(name)


def make_loader(
    ds: Dataset,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    sampler=None,
    device: torch.device | None = None,
) -> DataLoader:
    pin_memory = device is not None and device.type == "cuda"
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )


def probe_cfg_from_yaml(cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Subset of YAML saved in checkpoints so `eval/evaluate.py` can rebuild the same head."""
    return {"probe_embedding_batchnorm": bool(cfg.get("probe_embedding_batchnorm", False))}


def linear_probe_head_from_cfg(in_dim: int, num_classes: int, cfg: Mapping[str, Any]) -> nn.Module:
    from models.linear_probe import LinearProbeHead

    return LinearProbeHead(
        in_dim,
        num_classes=num_classes,
        embedding_batchnorm=bool(cfg.get("probe_embedding_batchnorm", False)),
    )


def balanced_class_weights(counts: torch.Tensor) -> torch.Tensor:
    """Inverse-frequency weights with mean 1 (sklearn-style balanced weights)."""
    c = counts.to(dtype=torch.float32).clamp(min=1.0)
    w = c.sum() / (c * c.numel())
    return w * (c.numel() / w.sum())


def build_probe_optimizer(params: Any, cfg: Mapping[str, Any]) -> Optimizer:
    kind = str(cfg.get("probe_optimizer", "sgd")).strip().lower()
    lr = float(cfg["lr"])
    wd = float(cfg.get("weight_decay", 0.0))
    if kind in {"adam", "adamw"}:
        return torch.optim.AdamW(params, lr=lr, weight_decay=wd)
    if kind == "sgd":
        mom = float(cfg.get("momentum", 0.9))
        return torch.optim.SGD(params, lr=lr, momentum=mom, weight_decay=wd)
    raise ValueError(f"Unknown probe_optimizer {kind!r}; use 'sgd' or 'adamw'.")


def build_probe_lr_scheduler(optimizer: Optimizer, cfg: Mapping[str, Any], total_epochs: int):
    name = str(cfg.get("lr_scheduler", "none")).strip().lower()
    if name in {"", "none"}:
        return None
    if name == "cosine":
        lr = float(cfg["lr"])
        eta_min = lr * float(cfg.get("lr_min_ratio", 0.05))
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(total_epochs), eta_min=eta_min)
    raise ValueError(f"Unknown lr_scheduler {name!r}; use 'none' or 'cosine'.")


def build_probe_criterion(
    num_classes: int,
    device: torch.device,
    cfg: Mapping[str, Any],
    *,
    class_counts: torch.Tensor | None,
) -> nn.CrossEntropyLoss:
    weight: torch.Tensor | None = None
    if bool(cfg.get("class_balanced_loss", False)) and class_counts is not None:
        weight = balanced_class_weights(class_counts).to(device)
    ls = float(cfg.get("label_smoothing", 0.0))
    return nn.CrossEntropyLoss(weight=weight, label_smoothing=ls)
