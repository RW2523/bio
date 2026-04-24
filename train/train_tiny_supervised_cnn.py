#!/usr/bin/env python3
"""Minimal end-to-end 1D CNN on WISDM windows (sanity / ceiling vs frozen probes)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.wisdm_supervised_dataset import WISDMSupervisedDataset
from eval.metrics import compute_metrics
from train.common import load_label_map, load_norm_stats, load_splits, make_loader, resolve_compute_device, resolve_path, subject_split_ids, to_bct
from utils.assertions import assert_disjoint_subject_sets, assert_label_range
from utils.checkpoint import load_checkpoint, save_checkpoint
from utils.io import read_json, write_json
from utils.logger import setup_logger
from utils.training_curves import save_supervised_curves_png
from utils.seed import set_seed
from utils.yaml_config import load_merged_config


class TinyCNN1d(nn.Module):
    """[B,T,3] -> logits [B,C] via small Conv1d stack + global pool."""

    def __init__(self, window_samples: int, num_classes: int) -> None:
        super().__init__()
        self.f = nn.Sequential(
            nn.Conv1d(3, 32, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 64, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
        )
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x_btc: torch.Tensor) -> torch.Tensor:
        x = to_bct(x_btc)
        h = self.f(x)
        z = h.mean(dim=-1)
        return self.fc(z)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="tiny_cnn_supervised")
    args = ap.parse_args()

    cfg = load_merged_config(args.config)
    set_seed(int(cfg.get("seed", 42)))

    out_root = resolve_path(cfg["output_dir"])
    art_dir = out_root / str(cfg["artifacts_subdir"])
    cache_dir = out_root / str(cfg["cache_subdir"])
    ckpt_dir = out_root / str(cfg.get("checkpoint_subdir", "checkpoints/tiny_cnn_supervised"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(log_file=ckpt_dir / "train.log")
    device = resolve_compute_device(str(cfg.get("compute_device", "cuda")))

    splits = load_splits(art_dir)
    train_ids, val_ids, test_ids = subject_split_ids(splits)
    assert_disjoint_subject_sets(train_ids, val_ids, test_ids)

    label_map = load_label_map(art_dir)
    num_classes = int(label_map["num_classes"])
    norm = load_norm_stats(art_dir)

    train_ds = WISDMSupervisedDataset(cache_dir, train_ids, mean=norm.mean, std=norm.std)
    val_ds = WISDMSupervisedDataset(cache_dir, val_ids, mean=norm.mean, std=norm.std)
    test_ds = WISDMSupervisedDataset(cache_dir, test_ids, mean=norm.mean, std=norm.std)

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
    test_loader = make_loader(
        test_ds,
        batch_size=int(cfg["batch_size"]),
        shuffle=False,
        num_workers=int(cfg["num_workers"]),
        device=device,
    )

    window_samples = int(read_json(art_dir / "preprocess_run.json")["window_samples"])
    model = TinyCNN1d(window_samples, num_classes).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg["lr"]), weight_decay=float(cfg.get("weight_decay", 0.0)))
    crit = nn.CrossEntropyLoss()

    def run_epoch(loader: DataLoader, train: bool) -> tuple[float, float]:
        if train:
            model.train()
        else:
            model.eval()
        tot, n, correct, count = 0.0, 0, 0, 0
        ctx = torch.enable_grad() if train else torch.no_grad()
        with ctx:
            for x, y, _w in loader:
                x = x.to(device)
                y = y.to(device)
                assert_label_range(y, num_classes)
                logits = model(x)
                loss = crit(logits, y)
                if train:
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    opt.step()
                tot += float(loss.detach().cpu())
                n += 1
                pred = logits.argmax(dim=-1)
                correct += int((pred == y).sum().item())
                count += int(y.numel())
        return tot / max(n, 1), correct / max(count, 1)

    best_val_loss = float("inf")
    best_val_acc = -1.0
    best_val_loss_epoch = 0
    best_val_acc_epoch = 0
    epochs = int(cfg["epochs"])
    curves: dict[str, list[float]] = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = run_epoch(train_loader, train=True)
        va_loss, va_acc = run_epoch(val_loader, train=False)
        curves["train_loss"].append(tr_loss)
        curves["val_loss"].append(va_loss)
        curves["train_acc"].append(tr_acc)
        curves["val_acc"].append(va_acc)
        logger.info("epoch=%d train_loss=%.4f train_acc=%.4f val_loss=%.4f val_acc=%.4f", epoch, tr_loss, tr_acc, va_loss, va_acc)
        payload = {"epoch": epoch, "model": model.state_dict(), "optimizer": opt.state_dict(), "num_classes": num_classes, "window_samples": window_samples}
        save_checkpoint(ckpt_dir / "last.pt", payload)
        improved_loss = va_loss < best_val_loss
        improved_acc = va_acc > best_val_acc
        if improved_loss:
            best_val_loss = va_loss
            best_val_loss_epoch = epoch
            save_checkpoint(ckpt_dir / "best_val_loss.pt", payload)
        if improved_acc:
            best_val_acc = va_acc
            best_val_acc_epoch = epoch
            save_checkpoint(ckpt_dir / "best_val_acc.pt", payload)
        if improved_acc:
            save_checkpoint(ckpt_dir / "best.pt", payload)

    write_json(ckpt_dir / "curves.json", curves)
    write_json(
        ckpt_dir / "best_epochs.json",
        {
            "best_val_loss": best_val_loss,
            "best_val_loss_epoch": best_val_loss_epoch,
            "best_val_acc": best_val_acc,
            "best_val_acc_epoch": best_val_acc_epoch,
            "best_checkpoint_metric": "val_acc",
        },
    )
    save_supervised_curves_png(curves, ckpt_dir / "curves.png", title=str(cfg.get("experiment_name", "")))

    ck = load_checkpoint(ckpt_dir / "best.pt", map_location=device)
    model.load_state_dict(ck["model"])
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for x, y, _w in test_loader:
            x = x.to(device)
            pred = model(x).argmax(dim=-1).cpu().numpy().tolist()
            ys.extend(y.numpy().tolist())
            ps.extend(pred)
    m = compute_metrics(np.asarray(ys, dtype=np.int64), np.asarray(ps, dtype=np.int64), labels=list(range(num_classes)))
    write_json(ckpt_dir / "test_metrics.json", m)
    logger.info(
        "TEST acc=%.4f macro_f1=%.4f weighted_f1=%.4f kappa=%.4f",
        m["accuracy"],
        m["macro_f1"],
        m["weighted_f1"],
        m["cohen_kappa"],
    )


if __name__ == "__main__":
    main()
