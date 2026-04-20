#!/usr/bin/env python3
"""AugPred-style SSL pretraining (3 binary tasks) with weighted sampling on train subjects."""

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

from datasets.weighted_sampler import build_weighted_sampler
from datasets.wisdm_ssl_dataset import WISDMSSLDataset
from models.spiking_resnet1d import SpikingResNet1d
from models.ssl_heads import AugPredSSLHeads
from train.common import (
    load_label_map,
    load_norm_stats,
    load_splits,
    log_module_trainable,
    make_loader,
    resolve_path,
    to_bct,
)
from transforms.augpred import sample_arrow_of_time, sample_permutation, sample_time_warp
from utils.assertions import assert_disjoint_subject_sets
from utils.checkpoint import save_checkpoint
from utils.io import read_json, write_json
from utils.logger import setup_logger
from utils.seed import set_seed
from utils.yaml_config import load_merged_config


def _smoke_batch(backbone: nn.Module, heads: AugPredSSLHeads, device: torch.device, window_samples: int) -> None:
    b = 2
    c = 3
    x = torch.randn(b, window_samples, c, device=device)
    xb = to_bct(x)
    z = backbone(xb)
    assert z.ndim == 2 and z.shape[0] == b
    a, p, t = heads(z)
    assert a.shape == (b, 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="ssl")
    args = ap.parse_args()

    cfg = load_merged_config(args.config)
    set_seed(int(cfg.get("seed", 42)))

    out_root = resolve_path(cfg["output_dir"])
    art_dir = out_root / str(cfg["artifacts_subdir"])
    cache_dir = out_root / str(cfg["cache_subdir"])
    ckpt_dir = out_root / str(cfg.get("checkpoint_subdir", "checkpoints/ssl"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(log_file=ckpt_dir / "pretrain.log")
    device = torch.device(str(cfg.get("compute_device", "cpu")))

    splits = load_splits(art_dir)
    train_ids = [int(x) for x in splits["train"]]
    val_ids = [int(x) for x in splits["val"]]
    test_ids = [int(x) for x in splits["test"]]
    assert_disjoint_subject_sets(train_ids, val_ids, test_ids)

    norm = load_norm_stats(art_dir)
    label_map = load_label_map(art_dir)
    num_classes = int(label_map["num_classes"])
    _ = num_classes  # unused here but validates artifact presence

    train_ds = WISDMSSLDataset(cache_dir, train_ids, mean=norm.mean, std=norm.std)
    val_ds = WISDMSSLDataset(cache_dir, val_ids, mean=norm.mean, std=norm.std)

    sampler = build_weighted_sampler(
        train_ds.motions,
        power=float(cfg.get("motion_weight_power", 1.0)),
        eps=float(cfg.get("motion_weight_eps", 1e-3)),
    )
    train_loader = make_loader(
        train_ds,
        batch_size=int(cfg["batch_size"]),
        shuffle=False,
        num_workers=int(cfg["num_workers"]),
        sampler=sampler,
    )
    val_loader = make_loader(val_ds, batch_size=int(cfg["batch_size"]), shuffle=False, num_workers=int(cfg["num_workers"]))

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
    heads = AugPredSSLHeads(backbone.out_dim).to(device)

    _smoke_batch(backbone, heads, device, window_samples)
    logger.info("Smoke batch OK.")

    params = list(backbone.parameters()) + list(heads.parameters())
    opt = torch.optim.Adam(params, lr=float(cfg["lr"]), weight_decay=float(cfg.get("weight_decay", 0.0)))
    crit = nn.BCEWithLogitsLoss()

    best_val = float("inf")
    curves: dict[str, list[float]] = {"train_loss": [], "val_loss": []}

    def ssl_step(x_btc: torch.Tensor) -> torch.Tensor:
        # three equal-weight tasks (3 forwards)
        x_btc = x_btc.to(device)
        losses = []
        # AOT
        xa, ya = sample_arrow_of_time(x_btc)
        za = backbone(to_bct(xa))
        la = crit(heads.head_aot(za).squeeze(-1), ya)
        losses.append(la)
        # Perm
        xp, yp = sample_permutation(x_btc, int(cfg.get("perm_num_chunks", 8)))
        zp = backbone(to_bct(xp))
        lp = crit(heads.head_perm(zp).squeeze(-1), yp)
        losses.append(lp)
        # Time warp
        xw, yw = sample_time_warp(x_btc, float(cfg.get("timewarp_sigma", 0.2)))
        zw = backbone(to_bct(xw))
        lw = crit(heads.head_tw(zw).squeeze(-1), yw)
        losses.append(lw)
        return la + lp + lw

    epochs = int(cfg["epochs"])
    for epoch in range(1, epochs + 1):
        backbone.train()
        heads.train()
        total = 0.0
        n = 0
        for x, _w in train_loader:
            opt.zero_grad(set_to_none=True)
            loss = ssl_step(x)
            loss.backward()
            opt.step()
            total += float(loss.detach().cpu())
            n += 1
        train_loss = total / max(n, 1)
        curves["train_loss"].append(train_loss)

        backbone.eval()
        heads.eval()
        vtot = 0.0
        vn = 0
        with torch.no_grad():
            for x, _w in val_loader:
                loss = ssl_step(x)
                vtot += float(loss.detach().cpu())
                vn += 1
        val_loss = vtot / max(vn, 1)
        curves["val_loss"].append(val_loss)

        logger.info("epoch=%d train_loss=%.5f val_loss=%.5f", epoch, train_loss, val_loss)

        payload = {
            "epoch": epoch,
            "model": {"backbone": backbone.state_dict(), "heads": heads.state_dict()},
            "optimizer": opt.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
        }
        save_checkpoint(ckpt_dir / "last.pt", payload)
        save_checkpoint(
            ckpt_dir / "backbone_last.pt",
            {"state_dict": backbone.state_dict(), "epoch": epoch, "model_cfg": model_cfg},
        )
        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(ckpt_dir / "best.pt", payload)
            save_checkpoint(
                ckpt_dir / "backbone_best.pt",
                {"state_dict": backbone.state_dict(), "epoch": epoch, "model_cfg": model_cfg},
            )

    write_json(ckpt_dir / "curves.json", curves)
    log_module_trainable(logger, "backbone", backbone)
    log_module_trainable(logger, "heads", heads)


if __name__ == "__main__":
    main()
