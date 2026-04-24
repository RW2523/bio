"""Shared train/val steps for frozen-backbone SNN + linear probe (Case 1 / Case 2)."""

from __future__ import annotations

from typing import Any, Mapping

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from train.common import motion_weighted_cross_entropy, probe_cfg_from_yaml, to_bct
from utils.assertions import assert_backbone_no_stored_gradients, assert_label_range


@torch.no_grad()
def eval_frozen_backbone_probe(
    backbone: nn.Module,
    head: nn.Module,
    loader: DataLoader,
    crit: nn.Module,
    device: torch.device,
    *,
    motion_loss_weighting: bool = False,
    motion_loss_alpha: float = 1.0,
) -> tuple[float, float]:
    """Mean loss and accuracy on ``loader`` (backbone and head in eval)."""
    backbone.eval()
    head.eval()
    tot, n_batches = 0.0, 0
    correct, count = 0, 0
    for x, y, _w in loader:
        x = x.to(device)
        y = y.to(device)
        w_motion = _w.to(device)
        z = backbone(to_bct(x))
        logits = head(z)
        if motion_loss_weighting:
            loss = motion_weighted_cross_entropy(logits, y, w_motion, alpha=motion_loss_alpha, crit=crit)
        else:
            loss = crit(logits, y)
        tot += float(loss.detach().cpu())
        n_batches += 1
        pred = torch.argmax(logits, dim=-1)
        correct += int((pred == y).sum().item())
        count += int(y.numel())
    return tot / max(n_batches, 1), correct / max(count, 1)


def train_one_epoch_frozen_backbone_probe(
    backbone: nn.Module,
    head: nn.Module,
    train_loader: DataLoader,
    opt: Optimizer,
    crit: nn.Module,
    device: torch.device,
    num_classes: int,
    max_grad_norm: float,
    probe_grad_checked: bool,
    *,
    motion_loss_weighting: bool = False,
    motion_loss_alpha: float = 1.0,
) -> tuple[float, float, bool]:
    """
    One epoch: backbone frozen under ``torch.no_grad``, gradients on head only.

    Returns ``train_loss``, ``train_acc``, ``probe_grad_checked``.
    """
    backbone.eval()
    head.train()
    total = 0.0
    n_batches = 0
    train_correct = 0
    train_count = 0
    for x, y, _w in train_loader:
        x = x.to(device)
        y = y.to(device)
        w_motion = _w.to(device)
        assert_label_range(y, num_classes)
        opt.zero_grad(set_to_none=True)
        with torch.no_grad():
            z = backbone(to_bct(x))
        logits = head(z)
        if motion_loss_weighting:
            loss = motion_weighted_cross_entropy(logits, y, w_motion, alpha=motion_loss_alpha, crit=crit)
        else:
            loss = crit(logits, y)
        loss.backward()
        if not probe_grad_checked:
            assert_backbone_no_stored_gradients(backbone)
            if not any(p.grad is not None and float(p.grad.detach().abs().sum()) > 0.0 for p in head.parameters()):
                raise AssertionError("Expected non-zero gradients on linear probe parameters.")
            probe_grad_checked = True
        if max_grad_norm > 0.0:
            torch.nn.utils.clip_grad_norm_(head.parameters(), max_grad_norm)
        opt.step()
        total += float(loss.detach().cpu())
        n_batches += 1
        pred = torch.argmax(logits.detach(), dim=-1)
        train_correct += int((pred == y).sum().item())
        train_count += int(y.numel())
    train_loss = total / max(n_batches, 1)
    train_acc = train_correct / max(train_count, 1)
    return train_loss, train_acc, probe_grad_checked


def snn_probe_checkpoint_payload(
    epoch: int,
    backbone: nn.Module,
    head: nn.Module,
    opt: Optimizer,
    model_cfg: Mapping[str, Any],
    num_classes: int,
    cfg: Mapping[str, Any],
    best_ckpt_metric: str,
    *,
    probe_cfg_saved: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Standard keys written by Case 1 / Case 2 linear-probe trainers."""
    pc = dict(probe_cfg_saved) if probe_cfg_saved is not None else probe_cfg_from_yaml(cfg)
    return {
        "epoch": epoch,
        "backbone": backbone.state_dict(),
        "head": head.state_dict(),
        "optimizer": opt.state_dict(),
        "model_cfg": model_cfg,
        "backbone_type": "spiking_resnet1d",
        "num_classes": num_classes,
        "probe_cfg": pc,
        "best_checkpoint_metric": best_ckpt_metric,
    }
