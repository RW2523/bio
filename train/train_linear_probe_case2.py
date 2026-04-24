#!/usr/bin/env python3
"""Case 2: AugPred-pretrained frozen spiking ResNet + linear probe."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.wisdm_supervised_dataset import WISDMSupervisedDataset
from train.common import (
    assert_optimizer_excludes_module,
    build_probe_criterion,
    build_probe_lr_scheduler,
    build_probe_optimizer,
    freeze_module,
    linear_probe_head_from_cfg,
    load_label_map,
    load_norm_stats,
    load_splits,
    log_module_trainable,
    make_loader,
    probe_cfg_from_yaml,
    resolve_compute_device,
    resolve_path,
    subject_split_ids,
    to_bct,
)
from train.frozen_linear_probe_loop import (
    eval_frozen_backbone_probe,
    snn_probe_checkpoint_payload,
    train_one_epoch_frozen_backbone_probe,
)
from train.snn_common import build_snn_backbone, merge_snn_model_cfg_from_checkpoint, snn_model_cfg_from_yaml
from utils.assertions import assert_backbone_frozen, assert_disjoint_subject_sets, assert_label_range
from utils.checkpoint import load_checkpoint, save_checkpoint
from utils.io import read_json, write_json
from utils.logger import setup_logger
from utils.training_curves import save_supervised_curves_png
from utils.seed import set_seed
from utils.yaml_config import load_merged_config


@torch.no_grad()
def _smoke(
    backbone: nn.Module,
    head: nn.Module,
    device: torch.device,
    window_samples: int,
    num_classes: int,
    in_channels: int,
) -> None:
    b = 2
    x = torch.randn(b, window_samples, int(in_channels), device=device)
    z = backbone(to_bct(x))
    logits = head(z)
    assert logits.shape == (b, num_classes)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="case2_probe")
    args = ap.parse_args()

    cfg = load_merged_config(args.config)
    set_seed(int(cfg.get("seed", 42)))

    out_root = resolve_path(cfg["output_dir"])
    art_dir = out_root / str(cfg["artifacts_subdir"])
    cache_dir = out_root / str(cfg["cache_subdir"])
    ckpt_dir = out_root / str(cfg.get("checkpoint_subdir", "checkpoints/case2"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(log_file=ckpt_dir / "train.log")
    device = resolve_compute_device(str(cfg.get("compute_device", "cuda")))
    logger.info("Compute device: %s", device)

    bb_path = cfg.get("pretrained_backbone_path")
    if not bb_path:
        raise ValueError("case2 requires `pretrained_backbone_path` in config (backbone-only SSL checkpoint).")
    bb_path = resolve_path(str(bb_path))
    if not bb_path.exists():
        raise FileNotFoundError(f"Missing pretrained backbone checkpoint: {bb_path}")

    splits = load_splits(art_dir)
    train_ids, val_ids, test_ids = subject_split_ids(splits)
    assert_disjoint_subject_sets(train_ids, val_ids, test_ids)

    label_map = load_label_map(art_dir)
    num_classes = int(label_map["num_classes"])
    norm = load_norm_stats(art_dir)

    yaml_model_cfg = snn_model_cfg_from_yaml(cfg)
    bb_ckpt = load_checkpoint(bb_path, map_location=device)
    model_cfg = merge_snn_model_cfg_from_checkpoint(bb_ckpt.get("model_cfg"), cfg)
    if bb_ckpt.get("model_cfg") is not None and bb_ckpt["model_cfg"] != yaml_model_cfg:
        logger.warning("Using `model_cfg` embedded in SSL backbone checkpoint (merged with current YAML defaults).")

    probe_yaml = probe_cfg_from_yaml(cfg)
    probe_yaml["feature_stack"] = list(model_cfg["feature_stack"])

    train_ds = WISDMSupervisedDataset(
        cache_dir, train_ids, mean=norm.mean, std=norm.std, feature_stack=model_cfg["feature_stack"]
    )
    val_ds = WISDMSupervisedDataset(
        cache_dir, val_ids, mean=norm.mean, std=norm.std, feature_stack=model_cfg["feature_stack"]
    )

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

    backbone = build_snn_backbone(model_cfg).to(device)

    backbone.load_state_dict(bb_ckpt["state_dict"], strict=True)
    logger.info("Loaded pretrained backbone weights from %s (epoch=%s)", bb_path, str(bb_ckpt.get("epoch")))

    freeze_module(backbone)
    assert_backbone_frozen(backbone)

    head = linear_probe_head_from_cfg(backbone.out_dim, num_classes, probe_yaml).to(device)

    _smoke(backbone, head, device, window_samples, num_classes, int(model_cfg["in_channels"]))
    logger.info("Smoke batch OK.")

    log_module_trainable(logger, "backbone", backbone)
    log_module_trainable(logger, "head", head)

    class_counts_t: torch.Tensor | None = None
    if bool(cfg.get("class_balanced_loss", False)):
        class_counts_t = torch.from_numpy(train_ds.count_labels(num_classes))
        logger.info(
            "Class-balanced loss: train window counts min/median/max = %d / %.0f / %d",
            int(class_counts_t.min()),
            float(class_counts_t.float().median()),
            int(class_counts_t.max()),
        )

    crit = build_probe_criterion(num_classes, device, cfg, class_counts=class_counts_t)
    opt = build_probe_optimizer(head.parameters(), cfg)
    assert_optimizer_excludes_module(opt, backbone)
    scheduler = build_probe_lr_scheduler(opt, cfg, int(cfg["epochs"]))
    if scheduler is not None:
        logger.info("Using lr_scheduler=%s on probe", str(cfg.get("lr_scheduler", "none")))

    max_grad_norm = float(cfg.get("max_grad_norm", 0.0))

    best_ckpt_metric = str(cfg.get("best_checkpoint_metric", "val_loss")).strip().lower()
    if best_ckpt_metric not in {"val_loss", "val_acc"}:
        raise ValueError("best_checkpoint_metric must be 'val_loss' or 'val_acc'")
    logger.info("Saving best.pt when %s improves (lower loss or higher acc).", best_ckpt_metric)
    motion_w = bool(cfg.get("motion_loss_weighting", False))
    motion_alpha = float(cfg.get("motion_loss_alpha", 1.0))
    logger.info("Motion-aware CE (frozen probe): enabled=%s alpha=%.4f", motion_w, motion_alpha)

    best_val_loss = float("inf")
    best_val_acc = -1.0
    best_val_loss_epoch = 0
    best_val_acc_epoch = 0
    curves: dict[str, list[float]] = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    epochs = int(cfg["epochs"])
    probe_grad_checked = False
    for epoch in range(1, epochs + 1):
        train_loss, train_acc, probe_grad_checked = train_one_epoch_frozen_backbone_probe(
            backbone,
            head,
            train_loader,
            opt,
            crit,
            device,
            num_classes,
            max_grad_norm,
            probe_grad_checked,
            motion_loss_weighting=motion_w,
            motion_loss_alpha=motion_alpha,
        )
        val_loss, val_acc = eval_frozen_backbone_probe(
            backbone,
            head,
            val_loader,
            crit,
            device,
            motion_loss_weighting=motion_w,
            motion_loss_alpha=motion_alpha,
        )
        curves["train_loss"].append(train_loss)
        curves["val_loss"].append(val_loss)
        curves["train_acc"].append(train_acc)
        curves["val_acc"].append(val_acc)
        lr_now = float(opt.param_groups[0]["lr"])
        logger.info(
            "epoch=%d train_loss=%.5f train_acc=%.4f val_loss=%.5f val_acc=%.4f lr=%.2e",
            epoch,
            train_loss,
            train_acc,
            val_loss,
            val_acc,
            lr_now,
        )
        if scheduler is not None:
            scheduler.step()

        payload = snn_probe_checkpoint_payload(
            epoch,
            backbone,
            head,
            opt,
            model_cfg,
            num_classes,
            cfg,
            best_ckpt_metric,
            probe_cfg_saved=probe_yaml,
        )
        save_checkpoint(ckpt_dir / "last.pt", payload)
        improved_loss = val_loss < best_val_loss
        improved_acc = val_acc > best_val_acc
        if improved_loss:
            best_val_loss = val_loss
            best_val_loss_epoch = epoch
            save_checkpoint(ckpt_dir / "best_val_loss.pt", payload)
        if improved_acc:
            best_val_acc = val_acc
            best_val_acc_epoch = epoch
            save_checkpoint(ckpt_dir / "best_val_acc.pt", payload)
        if (best_ckpt_metric == "val_acc" and improved_acc) or (best_ckpt_metric == "val_loss" and improved_loss):
            save_checkpoint(ckpt_dir / "best.pt", payload)

    write_json(ckpt_dir / "curves.json", curves)
    write_json(
        ckpt_dir / "best_epochs.json",
        {
            "best_val_loss": best_val_loss,
            "best_val_loss_epoch": best_val_loss_epoch,
            "best_val_acc": best_val_acc,
            "best_val_acc_epoch": best_val_acc_epoch,
            "best_checkpoint_metric": best_ckpt_metric,
        },
    )
    save_supervised_curves_png(curves, ckpt_dir / "curves.png", title=str(cfg.get("experiment_name", "")))


if __name__ == "__main__":
    main()
