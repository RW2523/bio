#!/usr/bin/env python3
"""Case 1: randomly initialized frozen spiking ResNet + linear probe."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.wisdm_supervised_dataset import WISDMSupervisedDataset
from models.linear_probe import LinearProbeHead
from models.spiking_resnet1d import SpikingResNet1d
from train.common import (
    assert_optimizer_excludes_module,
    freeze_module,
    load_label_map,
    load_norm_stats,
    load_splits,
    log_module_trainable,
    make_loader,
    resolve_compute_device,
    resolve_path,
    subject_split_ids,
    to_bct,
)
from utils.assertions import assert_backbone_frozen, assert_backbone_no_stored_gradients, assert_disjoint_subject_sets, assert_label_range
from utils.checkpoint import save_checkpoint
from utils.io import read_json, write_json
from utils.logger import setup_logger
from utils.seed import set_seed
from utils.yaml_config import load_merged_config


@torch.no_grad()
def _smoke(backbone: nn.Module, head: nn.Module, device: torch.device, window_samples: int, num_classes: int) -> None:
    b = 2
    x = torch.randn(b, window_samples, 3, device=device)
    z = backbone(to_bct(x))
    logits = head(z)
    assert logits.shape == (b, num_classes)
    assert_label_range(torch.tensor([0, num_classes - 1]), num_classes)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="case1_probe")
    args = ap.parse_args()

    cfg = load_merged_config(args.config)
    set_seed(int(cfg.get("seed", 42)))

    out_root = resolve_path(cfg["output_dir"])
    art_dir = out_root / str(cfg["artifacts_subdir"])
    cache_dir = out_root / str(cfg["cache_subdir"])
    ckpt_dir = out_root / str(cfg.get("checkpoint_subdir", "checkpoints/case1"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(log_file=ckpt_dir / "train.log")
    device = resolve_compute_device(str(cfg.get("compute_device", "cuda")))
    logger.info("Compute device: %s", device)

    splits = load_splits(art_dir)
    train_ids, val_ids, test_ids = subject_split_ids(splits)
    assert_disjoint_subject_sets(train_ids, val_ids, test_ids)

    label_map = load_label_map(art_dir)
    num_classes = int(label_map["num_classes"])
    norm = load_norm_stats(art_dir)

    train_ds = WISDMSupervisedDataset(cache_dir, train_ids, mean=norm.mean, std=norm.std)
    val_ds = WISDMSupervisedDataset(cache_dir, val_ids, mean=norm.mean, std=norm.std)

    train_loader = make_loader(
        train_ds,
        batch_size=int(cfg["batch_size"]),
        shuffle=True,
        num_workers=int(cfg["num_workers"]),
        device=device,
    )
    val_loader = make_loader(
        val_ds,
        batch_size=int(cfg["batch_size"]),
        shuffle=False,
        num_workers=int(cfg["num_workers"]),
        device=device,
    )

    window_samples = int(read_json(art_dir / "preprocess_run.json")["window_samples"])

    model_cfg = {
        "in_channels": int(cfg.get("in_channels", 3)),
        "base_channels": int(cfg.get("base_channels", 32)),
        "layers": [int(x) for x in cfg.get("layers", [1, 1, 2])],
        "stem_kernel": int(cfg.get("stem_kernel", 7)),
        "lif_beta": float(cfg.get("lif_beta", 0.9)),
        "lif_threshold": float(cfg.get("lif_threshold", 1.0)),
        "surrogate_alpha": float(cfg.get("surrogate_alpha", 2.0)),
        "lif_reset": str(cfg.get("lif_reset", "subtract")),
    }
    backbone = SpikingResNet1d(
        in_channels=model_cfg["in_channels"],
        base_channels=model_cfg["base_channels"],
        layers=tuple(model_cfg["layers"]),
        stem_kernel=model_cfg["stem_kernel"],
        beta=model_cfg["lif_beta"],
        threshold=model_cfg["lif_threshold"],
        surrogate_alpha=model_cfg["surrogate_alpha"],
        reset=model_cfg["lif_reset"],
    ).to(device)

    freeze_module(backbone)
    assert_backbone_frozen(backbone)

    head = LinearProbeHead(backbone.out_dim, num_classes=num_classes).to(device)

    _smoke(backbone, head, device, window_samples, num_classes)
    logger.info("Smoke batch OK.")

    log_module_trainable(logger, "backbone", backbone)
    log_module_trainable(logger, "head", head)

    opt = torch.optim.SGD(head.parameters(), lr=float(cfg["lr"]), momentum=0.9, weight_decay=float(cfg.get("weight_decay", 0.0)))
    assert_optimizer_excludes_module(opt, backbone)
    crit = nn.CrossEntropyLoss()

    best_val = float("inf")
    curves: dict[str, list[float]] = {"train_loss": [], "val_loss": [], "val_acc": []}

    def eval_loader(loader: DataLoader) -> tuple[float, float]:
        backbone.eval()
        head.eval()
        tot = 0.0
        n = 0
        correct = 0
        count = 0
        with torch.no_grad():
            for x, y, _w in loader:
                x = x.to(device)
                y = y.to(device)
                z = backbone(to_bct(x))
                logits = head(z)
                loss = crit(logits, y)
                tot += float(loss.detach().cpu())
                n += 1
                pred = torch.argmax(logits, dim=-1)
                correct += int((pred == y).sum().item())
                count += int(y.numel())
        return tot / max(n, 1), correct / max(count, 1)

    epochs = int(cfg["epochs"])
    probe_grad_checked = False
    for epoch in range(1, epochs + 1):
        backbone.eval()
        head.train()
        total = 0.0
        m = 0
        for x, y, _w in train_loader:
            x = x.to(device)
            y = y.to(device)
            assert_label_range(y, num_classes)
            opt.zero_grad(set_to_none=True)
            with torch.no_grad():
                z = backbone(to_bct(x))
            logits = head(z)
            loss = crit(logits, y)
            loss.backward()
            if not probe_grad_checked:
                assert_backbone_no_stored_gradients(backbone)
                if not any(p.grad is not None and float(p.grad.detach().abs().sum()) > 0.0 for p in head.parameters()):
                    raise AssertionError("Expected non-zero gradients on linear probe parameters.")
                probe_grad_checked = True
            opt.step()
            total += float(loss.detach().cpu())
            m += 1
        train_loss = total / max(m, 1)
        val_loss, val_acc = eval_loader(val_loader)
        curves["train_loss"].append(train_loss)
        curves["val_loss"].append(val_loss)
        curves["val_acc"].append(val_acc)
        logger.info("epoch=%d train_loss=%.5f val_loss=%.5f val_acc=%.4f", epoch, train_loss, val_loss, val_acc)

        payload = {
            "epoch": epoch,
            "backbone": backbone.state_dict(),
            "head": head.state_dict(),
            "optimizer": opt.state_dict(),
            "model_cfg": model_cfg,
            "num_classes": num_classes,
        }
        save_checkpoint(ckpt_dir / "last.pt", payload)
        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(ckpt_dir / "best.pt", payload)

    write_json(ckpt_dir / "curves.json", curves)


if __name__ == "__main__":
    main()
