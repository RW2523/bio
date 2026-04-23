#!/usr/bin/env python3
"""AugPred-style SSL pretraining with a continuous CNN ResNet-1D backbone."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import WeightedRandomSampler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.weighted_sampler import build_weighted_sampler
from datasets.wisdm_ssl_dataset import WISDMSSLDataset
from models.ssl_heads import AugPredSSLHeads
from train.cnn_common import build_cnn_backbone, cnn_model_cfg_from_yaml
from train.common import (
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
from transforms.augpred import sample_arrow_of_time, sample_permutation, sample_time_warp
from utils.assertions import assert_disjoint_subject_sets
from utils.checkpoint import save_checkpoint
from utils.io import read_json, write_json
from utils.logger import setup_logger
from utils.seed import set_seed
from utils.yaml_config import load_merged_config


@torch.no_grad()
def _smoke_batch(backbone: nn.Module, heads: AugPredSSLHeads, device: torch.device, window_samples: int) -> None:
    b = 2
    x = torch.randn(b, window_samples, 3, device=device)
    z = backbone(to_bct(x))
    assert z.ndim == 2 and z.shape[0] == b
    a, p, t = heads(z)
    assert a.shape == p.shape == t.shape == (b, 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="ssl_cnn")
    args = ap.parse_args()

    cfg = load_merged_config(args.config)
    set_seed(int(cfg.get("seed", 42)))

    out_root = resolve_path(cfg["output_dir"])
    art_dir = out_root / str(cfg["artifacts_subdir"])
    cache_dir = out_root / str(cfg["cache_subdir"])
    ckpt_dir = out_root / str(cfg.get("checkpoint_subdir", "checkpoints/cnn_ssl"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(log_file=ckpt_dir / "pretrain.log")
    device = resolve_compute_device(str(cfg.get("compute_device", "cuda")))
    logger.info("Compute device: %s", device)

    splits = load_splits(art_dir)
    train_ids, val_ids, test_ids = subject_split_ids(splits)
    assert_disjoint_subject_sets(train_ids, val_ids, test_ids)

    norm = load_norm_stats(art_dir)
    label_map = load_label_map(art_dir)
    _ = int(label_map["num_classes"])

    train_ds = WISDMSSLDataset(cache_dir, train_ids, mean=norm.mean, std=norm.std)
    val_ds = WISDMSSLDataset(cache_dir, val_ids, mean=norm.mean, std=norm.std)

    sampler = build_weighted_sampler(
        train_ds.motions,
        power=float(cfg.get("motion_weight_power", 1.0)),
        eps=float(cfg.get("motion_weight_eps", 1e-3)),
    )
    if not isinstance(sampler, WeightedRandomSampler):
        raise TypeError("Expected WeightedRandomSampler for SSL training dataloader.")
    logger.info("Using WeightedRandomSampler on %d train windows.", len(train_ds))

    train_loader = make_loader(
        train_ds,
        batch_size=int(cfg["batch_size"]),
        shuffle=False,
        num_workers=int(cfg["num_workers"]),
        sampler=sampler,
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
    model_cfg = cnn_model_cfg_from_yaml(cfg)
    backbone = build_cnn_backbone(model_cfg).to(device)
    heads = AugPredSSLHeads(backbone.out_dim).to(device)

    _smoke_batch(backbone, heads, device, window_samples)
    logger.info("Smoke batch OK.")

    opt = torch.optim.Adam(
        list(backbone.parameters()) + list(heads.parameters()),
        lr=float(cfg["lr"]),
        weight_decay=float(cfg.get("weight_decay", 0.0)),
    )
    crit = nn.BCEWithLogitsLoss()
    best_val = float("inf")
    curves: dict[str, list[float]] = {"train_loss": [], "val_loss": []}

    def ssl_step(x_btc: torch.Tensor) -> torch.Tensor:
        x_btc = x_btc.to(device)
        xa, ya = sample_arrow_of_time(x_btc)
        la = crit(heads.head_aot(backbone(to_bct(xa))).squeeze(-1), ya)
        xp, yp = sample_permutation(x_btc, int(cfg.get("perm_num_chunks", 8)))
        lp = crit(heads.head_perm(backbone(to_bct(xp))).squeeze(-1), yp)
        xw, yw = sample_time_warp(x_btc, float(cfg.get("timewarp_sigma", 0.2)))
        lw = crit(heads.head_tw(backbone(to_bct(xw))).squeeze(-1), yw)
        return la + lp + lw

    for epoch in range(1, int(cfg["epochs"]) + 1):
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
                vtot += float(ssl_step(x).detach().cpu())
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
            {"state_dict": backbone.state_dict(), "epoch": epoch, "model_cfg": model_cfg, "backbone_type": "cnn_resnet1d"},
        )
        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(ckpt_dir / "best.pt", payload)
            save_checkpoint(
                ckpt_dir / "backbone_best.pt",
                {"state_dict": backbone.state_dict(), "epoch": epoch, "model_cfg": model_cfg, "backbone_type": "cnn_resnet1d"},
            )

    write_json(ckpt_dir / "curves.json", curves)
    log_module_trainable(logger, "backbone", backbone)
    log_module_trainable(logger, "heads", heads)


if __name__ == "__main__":
    main()
