#!/usr/bin/env python3
"""Case 2 variant: SSL backbone + linear head, backbone unfrozen for end-to-end fine-tuning."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.wisdm_supervised_dataset import WISDMSupervisedDataset
from eval.confusion_matrix import save_confusion_matrix_figure
from eval.metrics import compute_metrics
from train.common import (
    build_probe_criterion,
    build_probe_lr_scheduler,
    linear_probe_head_from_cfg,
    load_label_map,
    load_norm_stats,
    load_splits,
    log_module_trainable,
    make_loader,
    motion_weighted_cross_entropy,
    probe_cfg_from_yaml,
    resolve_compute_device,
    resolve_path,
    subject_split_ids,
    to_bct,
)
from train.snn_common import build_snn_backbone, merge_snn_model_cfg_from_checkpoint, snn_model_cfg_from_yaml
from train.ssl_debug import backbone_grad_norm_l2, read_spike_mean_from_snn, spike_mean_proxy_health_note
from utils.assertions import assert_disjoint_subject_sets, assert_label_range
from utils.checkpoint import load_checkpoint, save_checkpoint
from utils.io import read_json, write_json
from utils.logger import setup_logger
from utils.seed import set_seed
from utils.training_curves import save_supervised_curves_png
from utils.yaml_config import load_merged_config

_LOW_MOTION_SUBSTR = ("sitting", "standing", "typing", "writing", "eating")


def _log_low_motion_class_f1(logger, label_map: dict, per_class: dict, *, split: str = "val") -> None:
    """Log per-class F1 for low-motion activity names (sitting, standing, typing, writing, eating)."""
    code_to_name = label_map.get("code_to_name") or {}
    index_to_code: list[str] = list(label_map["index_to_code"])
    for i in range(len(index_to_code)):
        code = index_to_code[i]
        name = str(code_to_name.get(code, "")).lower()
        if not any(s in name for s in _LOW_MOTION_SUBSTR):
            continue
        row = per_class.get(str(i), {})
        f1v = row.get("f1-score", 0.0)
        logger.info(
            "%s per-class F1: idx=%d code=%s name=%s f1=%.4f",
            split,
            i,
            code,
            code_to_name.get(code, ""),
            float(f1v),
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="case2_e2e_finetune")
    args = ap.parse_args()

    cfg = load_merged_config(args.config)
    set_seed(int(cfg.get("seed", 42)))

    out_root = resolve_path(cfg["output_dir"])
    art_dir = out_root / str(cfg["artifacts_subdir"])
    cache_dir = out_root / str(cfg["cache_subdir"])
    ckpt_dir = out_root / str(cfg.get("checkpoint_subdir", "checkpoints/case2_e2e"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(log_file=ckpt_dir / "train.log")
    device = resolve_compute_device(str(cfg.get("compute_device", "cuda")))
    logger.info("Compute device: %s", device)

    bb_path = cfg.get("pretrained_backbone_path")
    if not bb_path:
        raise ValueError("pretrained_backbone_path required (SSL backbone_best.pt).")
    bb_path = resolve_path(str(bb_path))
    if not bb_path.exists():
        raise FileNotFoundError(bb_path)

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
        logger.warning("Merged SSL `model_cfg` with current YAML defaults for fine-tuning.")

    probe_yaml = probe_cfg_from_yaml(cfg)
    probe_yaml["feature_stack"] = list(model_cfg["feature_stack"])

    train_ds = WISDMSupervisedDataset(
        cache_dir, train_ids, mean=norm.mean, std=norm.std, feature_stack=model_cfg["feature_stack"]
    )
    val_ds = WISDMSupervisedDataset(
        cache_dir, val_ids, mean=norm.mean, std=norm.std, feature_stack=model_cfg["feature_stack"]
    )
    test_ds = WISDMSupervisedDataset(
        cache_dir, test_ids, mean=norm.mean, std=norm.std, feature_stack=model_cfg["feature_stack"]
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
    test_loader = make_loader(
        test_ds,
        batch_size=int(cfg["batch_size"]),
        shuffle=False,
        num_workers=int(cfg["num_workers"]),
        device=device,
    )

    window_samples = int(read_json(art_dir / "preprocess_run.json")["window_samples"])
    backbone = build_snn_backbone(model_cfg).to(device)
    backbone.load_state_dict(bb_ckpt["state_dict"], strict=True)

    finetune = cfg.get("finetune") if isinstance(cfg.get("finetune"), dict) else {}
    unfreeze_bb = bool(finetune.get("unfreeze_backbone", cfg.get("finetune_unfreeze_backbone", True)))
    motion_w = bool(finetune.get("motion_loss_weighting", cfg.get("motion_loss_weighting", False)))
    motion_alpha = float(finetune.get("motion_loss_alpha", cfg.get("motion_loss_alpha", 1.0)))

    logger.info(
        "E2E options: pooling=%s pool_bins=%s classifier_head=%s motion_loss_weighting=%s "
        "motion_loss_alpha=%.4f unfreeze_backbone=%s",
        model_cfg.get("pooling"),
        model_cfg.get("pool_bins"),
        probe_yaml.get("classifier_head"),
        motion_w,
        motion_alpha,
        unfreeze_bb,
    )

    if unfreeze_bb:
        for p in backbone.parameters():
            p.requires_grad = True
        backbone.train()
    else:
        for p in backbone.parameters():
            p.requires_grad = False
        backbone.eval()
        logger.info("Backbone frozen (classifier-only fine-tune).")

    head = linear_probe_head_from_cfg(backbone.out_dim, num_classes, probe_yaml).to(device)

    in_ch = int(model_cfg["in_channels"])
    b = 2
    x0 = torch.randn(b, window_samples, in_ch, device=device)
    logits0 = head(backbone(to_bct(x0)))
    assert logits0.shape == (b, num_classes)
    logger.info("Smoke batch OK (in_channels=%d).", in_ch)

    log_module_trainable(logger, "backbone", backbone)
    log_module_trainable(logger, "head", head)

    class_counts_t: torch.Tensor | None = None
    if bool(cfg.get("class_balanced_loss", False)):
        class_counts_t = torch.from_numpy(train_ds.count_labels(num_classes))

    crit = build_probe_criterion(num_classes, device, cfg, class_counts=class_counts_t)
    head_lr = float(finetune.get("head_lr", cfg.get("head_lr", cfg.get("lr", 0.001))))
    bb_lr = float(
        finetune.get("backbone_lr", cfg.get("backbone_lr", head_lr * float(cfg.get("backbone_lr_ratio", 0.1))))
    )
    wd = float(finetune.get("weight_decay", cfg.get("weight_decay", 0.0)))
    if unfreeze_bb:
        opt = torch.optim.AdamW(
            [{"params": backbone.parameters(), "lr": bb_lr}, {"params": head.parameters(), "lr": head_lr}],
            weight_decay=wd,
        )
        logger.info("Optimizer LRs: backbone=%.2e head=%.2e weight_decay=%.2e", bb_lr, head_lr, wd)
    else:
        opt = torch.optim.AdamW(head.parameters(), lr=head_lr, weight_decay=wd)
        logger.info("Optimizer (head only): lr=%.2e weight_decay=%.2e", head_lr, wd)

    scheduler = build_probe_lr_scheduler(opt, cfg, int(cfg["epochs"]))
    max_grad_norm = float(cfg.get("max_grad_norm", 0.0))

    best_ckpt_metric = str(cfg.get("best_checkpoint_metric", "val_loss")).strip().lower()
    if str(finetune.get("early_stop_metric", "")).strip().lower() == "macro_f1":
        best_ckpt_metric = "val_macro_f1"
    if best_ckpt_metric not in {"val_loss", "val_acc", "val_macro_f1"}:
        raise ValueError("best_checkpoint_metric must be 'val_loss', 'val_acc', or 'val_macro_f1'")

    early_patience = int(cfg.get("early_stop_patience", 0))
    epochs = int(cfg["epochs"])
    dbg = cfg.get("debug") or {}
    dbg = dbg if isinstance(dbg, dict) else {}
    log_gn = bool(dbg.get("log_grad_norm", False))
    log_sp = bool(dbg.get("log_spike_rate", False))

    best_val_loss = float("inf")
    best_val_acc = -1.0
    best_val_macro_f1 = -1.0
    best_val_loss_epoch = 0
    best_val_acc_epoch = 0
    best_val_macro_f1_epoch = 0
    curves: dict[str, list[float]] = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "val_macro_f1": [],
        "val_weighted_f1": [],
    }
    labels = list(range(num_classes))
    stall = 0

    def eval_epoch(loader: DataLoader) -> tuple[float, float, float, float, dict]:
        backbone.eval()
        head.eval()
        tot = 0.0
        n_batches = 0
        correct = 0
        count = 0
        ys: list[int] = []
        ps: list[int] = []
        with torch.no_grad():
            for x, y, _w in loader:
                x = x.to(device)
                y = y.to(device)
                w_motion = _w.to(device)
                logits = head(backbone(to_bct(x)))
                if motion_w:
                    loss = motion_weighted_cross_entropy(
                        logits, y, w_motion, alpha=motion_alpha, crit=crit
                    )
                else:
                    loss = crit(logits, y)
                tot += float(loss.detach().cpu())
                n_batches += 1
                pred = torch.argmax(logits, dim=-1)
                correct += int((pred == y).sum().item())
                count += int(y.numel())
                ys.extend(y.detach().cpu().numpy().tolist())
                ps.extend(pred.detach().cpu().numpy().tolist())
        y_true = np.asarray(ys, dtype=np.int64)
        y_pred = np.asarray(ps, dtype=np.int64)
        metrics = compute_metrics(y_true, y_pred, labels=labels)
        val_loss = tot / max(n_batches, 1)
        val_acc = correct / max(count, 1)
        macro = float(metrics["macro_f1"])
        w_f1 = float(metrics["weighted_f1"])
        return val_loss, val_acc, macro, w_f1, metrics

    for epoch in range(1, epochs + 1):
        if unfreeze_bb:
            backbone.train()
        else:
            backbone.eval()
        head.train()
        total = 0.0
        m = 0
        train_correct = 0
        train_count = 0
        max_gn = 0.0
        for x, y, _w in train_loader:
            x = x.to(device)
            y = y.to(device)
            w_motion = _w.to(device)
            assert_label_range(y, num_classes)
            opt.zero_grad(set_to_none=True)
            logits = head(backbone(to_bct(x)))
            if motion_w:
                loss = motion_weighted_cross_entropy(
                    logits, y, w_motion, alpha=motion_alpha, crit=crit
                )
            else:
                loss = crit(logits, y)
            loss.backward()
            if log_gn and unfreeze_bb:
                max_gn = max(max_gn, backbone_grad_norm_l2(backbone))
            if max_grad_norm > 0.0:
                to_clip = (
                    list(head.parameters())
                    if not unfreeze_bb
                    else list(backbone.parameters()) + list(head.parameters())
                )
                torch.nn.utils.clip_grad_norm_(to_clip, max_grad_norm)
            opt.step()
            total += float(loss.detach().cpu())
            m += 1
            pred = torch.argmax(logits.detach(), dim=-1)
            train_correct += int((pred == y).sum().item())
            train_count += int(y.numel())

        train_loss = total / max(m, 1)
        train_acc = train_correct / max(train_count, 1)
        val_loss, val_acc, val_macro_f1, val_wf1, val_metrics = eval_epoch(val_loader)
        curves["train_loss"].append(train_loss)
        curves["val_loss"].append(val_loss)
        curves["train_acc"].append(train_acc)
        curves["val_acc"].append(val_acc)
        curves["val_macro_f1"].append(val_macro_f1)
        curves["val_weighted_f1"].append(val_wf1)

        logger.info(
            "epoch=%d train_loss=%.5f train_acc=%.4f val_loss=%.5f val_acc=%.4f val_macro_f1=%.4f val_weighted_f1=%.4f",
            epoch,
            train_loss,
            train_acc,
            val_loss,
            val_acc,
            val_macro_f1,
            val_wf1,
        )
        _log_low_motion_class_f1(logger, label_map, val_metrics["per_class_report"], split="val")
        if log_gn:
            if unfreeze_bb:
                logger.info("epoch=%d backbone_grad_norm_max=%.6f", epoch, max_gn)
            else:
                logger.info("epoch=%d backbone_grad_norm skipped (backbone frozen)", epoch)
        if log_sp:
            if unfreeze_bb:
                backbone.train()
            x0b, _, _ = next(iter(val_loader))
            x0b = x0b.to(device)
            mnb = min(2, int(x0b.shape[0]))
            _ = backbone(to_bct(x0b[:mnb]))
            sm = read_spike_mean_from_snn(backbone)
            logger.info(
                "epoch=%d conv_spike_mean_proxy=%s%s",
                epoch,
                f"{sm:.6f}" if sm is not None else "n/a",
                spike_mean_proxy_health_note(sm),
            )

        if scheduler is not None:
            scheduler.step()

        payload = {
            "epoch": epoch,
            "backbone": backbone.state_dict(),
            "head": head.state_dict(),
            "optimizer": opt.state_dict(),
            "model_cfg": model_cfg,
            "backbone_type": "spiking_resnet1d",
            "num_classes": num_classes,
            "probe_cfg": probe_yaml,
            "finetune": True,
            "finetune_unfreeze_backbone": unfreeze_bb,
            "best_checkpoint_metric": best_ckpt_metric,
        }
        save_checkpoint(ckpt_dir / "last.pt", payload)
        improved_loss = val_loss < best_val_loss
        improved_acc = val_acc > best_val_acc
        improved_f1 = val_macro_f1 > best_val_macro_f1
        if improved_loss:
            best_val_loss = val_loss
            best_val_loss_epoch = epoch
            save_checkpoint(ckpt_dir / "best_val_loss.pt", payload)
        if improved_acc:
            best_val_acc = val_acc
            best_val_acc_epoch = epoch
            save_checkpoint(ckpt_dir / "best_val_acc.pt", payload)
        if improved_f1:
            best_val_macro_f1 = val_macro_f1
            best_val_macro_f1_epoch = epoch
            save_checkpoint(ckpt_dir / "best_val_macro_f1.pt", payload)
        if (best_ckpt_metric == "val_acc" and improved_acc) or (
            best_ckpt_metric == "val_loss" and improved_loss
        ) or (best_ckpt_metric == "val_macro_f1" and improved_f1):
            save_checkpoint(ckpt_dir / "best.pt", payload)

        if early_patience > 0 and best_ckpt_metric == "val_macro_f1":
            if improved_f1:
                stall = 0
            else:
                stall += 1
                if stall >= early_patience:
                    logger.info("Early stopping: no val_macro_f1 improvement for %d epochs.", early_patience)
                    break

    write_json(ckpt_dir / "curves.json", curves)
    write_json(
        ckpt_dir / "best_epochs.json",
        {
            "best_val_loss": best_val_loss,
            "best_val_loss_epoch": best_val_loss_epoch,
            "best_val_acc": best_val_acc,
            "best_val_acc_epoch": best_val_acc_epoch,
            "best_val_macro_f1": best_val_macro_f1,
            "best_val_macro_f1_epoch": best_val_macro_f1_epoch,
            "best_checkpoint_metric": best_ckpt_metric,
        },
    )
    save_supervised_curves_png(curves, ckpt_dir / "curves.png", title=str(cfg.get("experiment_name", "")))

    best_path = ckpt_dir / "best.pt"
    if not best_path.is_file():
        logger.warning("No best.pt written; skip held-out test export.")
    else:
        ck_best = load_checkpoint(best_path, map_location=device)
        backbone.load_state_dict(ck_best["backbone"], strict=True)
        head.load_state_dict(ck_best["head"], strict=True)
        backbone.eval()
        head.eval()
        ys_t: list[int] = []
        ps_t: list[int] = []
        with torch.no_grad():
            for x, y, _w in test_loader:
                x = x.to(device)
                logits = head(backbone(to_bct(x)))
                pred = torch.argmax(logits, dim=-1).detach().cpu().numpy().tolist()
                ys_t.extend(y.detach().cpu().numpy().tolist())
                ps_t.extend(pred)
        y_true_t = np.asarray(ys_t, dtype=np.int64)
        y_pred_t = np.asarray(ps_t, dtype=np.int64)
        test_metrics = compute_metrics(y_true_t, y_pred_t, labels=labels)
        test_dir = ckpt_dir / "test_eval"
        test_dir.mkdir(parents=True, exist_ok=True)
        metrics_json = {k: v for k, v in test_metrics.items() if k != "per_class_report"}
        write_json(test_dir / "metrics.json", metrics_json)
        index_to_code: list[str] = list(label_map["index_to_code"])
        class_names = [f"{i}:{index_to_code[i]}" for i in range(num_classes)]
        report_txt = classification_report(
            y_true_t,
            y_pred_t,
            labels=labels,
            target_names=class_names,
            digits=4,
            zero_division=0,
        )
        (test_dir / "classification_report.txt").write_text(report_txt, encoding="utf-8")
        cm = np.asarray(test_metrics["confusion_matrix"], dtype=np.int64)
        save_confusion_matrix_figure(
            cm,
            class_names=class_names,
            out_path=test_dir / "confusion_matrix.png",
            title="Confusion matrix (test, best checkpoint)",
        )
        row = {
            "macro_f1": test_metrics["macro_f1"],
            "weighted_f1": test_metrics["weighted_f1"],
            "accuracy": test_metrics["accuracy"],
            "cohen_kappa": test_metrics["cohen_kappa"],
        }
        with (test_dir / "metrics_summary.csv").open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row.keys()))
            w.writeheader()
            w.writerow(row)
        logger.info(
            "TEST (best.pt) acc=%.4f macro_f1=%.4f weighted_f1=%.4f kappa=%.4f — wrote %s",
            test_metrics["accuracy"],
            test_metrics["macro_f1"],
            test_metrics["weighted_f1"],
            test_metrics["cohen_kappa"],
            test_dir,
        )
        _log_low_motion_class_f1(logger, label_map, test_metrics["per_class_report"], split="test")


if __name__ == "__main__":
    main()
