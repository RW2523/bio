#!/usr/bin/env python3
"""AugPred-style SSL pretraining (3 binary tasks) with weighted sampling on train subjects."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.weighted_sampler import build_weighted_sampler
from datasets.wisdm_ssl_dataset import WISDMSSLDataset
from models.ssl_heads import AugPredSSLHeads
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
from train.snn_common import build_snn_backbone, snn_model_cfg_from_yaml
from transforms.augpred import sample_arrow_of_time, sample_permutation, sample_time_warp
from utils.assertions import assert_disjoint_subject_sets
from utils.checkpoint import save_checkpoint
from utils.io import read_json, write_json
from utils.logger import setup_logger
from utils.training_curves import save_ssl_loss_curves_png
from utils.seed import set_seed
from utils.yaml_config import load_merged_config

from train.ssl_debug import backbone_grad_norm_l2, read_spike_mean_from_snn, spike_mean_proxy_health_note


def _smoke_batch(backbone: nn.Module, heads: AugPredSSLHeads, device: torch.device, window_samples: int, in_channels: int) -> None:
    b = 2
    c = int(in_channels)
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
    device = resolve_compute_device(str(cfg.get("compute_device", "cuda")))
    logger.info("Compute device: %s", device)

    splits = load_splits(art_dir)
    train_ids, val_ids, test_ids = subject_split_ids(splits)
    assert_disjoint_subject_sets(train_ids, val_ids, test_ids)

    norm = load_norm_stats(art_dir)
    label_map = load_label_map(art_dir)
    num_classes = int(label_map["num_classes"])
    _ = num_classes  # unused here but validates artifact presence

    model_cfg = snn_model_cfg_from_yaml(cfg)
    train_ds = WISDMSSLDataset(
        cache_dir, train_ids, mean=norm.mean, std=norm.std, feature_stack=model_cfg["feature_stack"]
    )
    val_ds = WISDMSSLDataset(
        cache_dir, val_ids, mean=norm.mean, std=norm.std, feature_stack=model_cfg["feature_stack"]
    )

    sampler = build_weighted_sampler(
        train_ds.motions,
        power=float(cfg.get("motion_weight_power", 1.0)),
        eps=float(cfg.get("motion_weight_eps", 1e-3)),
    )
    if not isinstance(sampler, WeightedRandomSampler):
        raise TypeError("Expected WeightedRandomSampler for SSL training dataloader.")
    logger.info(
        "Using WeightedRandomSampler on %d train windows (motion^%.3f + eps=%.1e).",
        len(train_ds),
        float(cfg.get("motion_weight_power", 1.0)),
        float(cfg.get("motion_weight_eps", 1e-3)),
    )
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

    backbone = build_snn_backbone(model_cfg).to(device)
    heads = AugPredSSLHeads(backbone.out_dim).to(device)

    _smoke_batch(backbone, heads, device, window_samples, int(model_cfg["in_channels"]))
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

    dbg = cfg.get("debug") or {}
    dbg = dbg if isinstance(dbg, dict) else {}
    log_gn = bool(dbg.get("log_grad_norm", False))
    log_sp = bool(dbg.get("log_spike_rate", False))

    epochs = int(cfg["epochs"])
    for epoch in range(1, epochs + 1):
        backbone.train()
        heads.train()
        total = 0.0
        n = 0
        max_gn = 0.0
        for x, _w in train_loader:
            opt.zero_grad(set_to_none=True)
            loss = ssl_step(x)
            loss.backward()
            if log_gn:
                max_gn = max(max_gn, backbone_grad_norm_l2(backbone))
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
        if log_gn:
            logger.info("epoch=%d backbone_grad_norm_max=%.6f", epoch, max_gn)
        if log_sp:
            backbone.train()
            x0, _ = next(iter(val_loader))
            x0 = x0.to(device)
            m = min(2, int(x0.shape[0]))
            _ = backbone(to_bct(x0[:m]))
            sm = read_spike_mean_from_snn(backbone)
            logger.info(
                "epoch=%d conv_spike_mean_proxy=%s%s",
                epoch,
                f"{sm:.6f}" if sm is not None else "n/a",
                spike_mean_proxy_health_note(sm),
            )

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
            {"state_dict": backbone.state_dict(), "epoch": epoch, "model_cfg": model_cfg, "backbone_type": "spiking_resnet1d"},
        )
        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(ckpt_dir / "best.pt", payload)
            save_checkpoint(
                ckpt_dir / "backbone_best.pt",
                {"state_dict": backbone.state_dict(), "epoch": epoch, "model_cfg": model_cfg, "backbone_type": "spiking_resnet1d"},
            )

    write_json(ckpt_dir / "curves.json", curves)
    save_ssl_loss_curves_png(curves, ckpt_dir / "curves.png", title=str(cfg.get("experiment_name", "")))
    log_module_trainable(logger, "backbone", backbone)
    log_module_trainable(logger, "heads", heads)


if __name__ == "__main__":
    main()
