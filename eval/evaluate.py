#!/usr/bin/env python3
"""Evaluate a trained linear-probe checkpoint on held-out test subjects."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.wisdm_supervised_dataset import WISDMSupervisedDataset
from eval.confusion_matrix import save_confusion_matrix_figure
from eval.metrics import compute_metrics
from sklearn.metrics import classification_report

from train.common import (
    freeze_module,
    infer_artifacts_dir_from_checkpoint,
    linear_probe_head_from_cfg,
    load_label_map,
    load_norm_stats,
    load_splits,
    probe_cfg_from_checkpoint,
    resolve_compute_device,
    subject_split_ids,
    to_bct,
)
from train.snn_common import build_snn_backbone, merge_snn_model_cfg_from_checkpoint
from utils.assertions import assert_backbone_frozen, assert_disjoint_subject_sets
from utils.checkpoint import load_checkpoint
from utils.io import write_json
from utils.logger import setup_logger
from utils.paths import project_root
from utils.yaml_config import load_merged_config


def _resolve(p: str) -> Path:
    pp = Path(p)
    return pp if pp.is_absolute() else (project_root() / pp).resolve()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Probe checkpoint (`best.pt`, or `best_val_acc.pt` / `best_val_loss.pt` after training)",
    )
    ap.add_argument(
        "--artifacts_dir",
        type=str,
        default="",
        help="Directory containing label_map.json/norm_stats.json/splits.json/preprocess_run.json (default: inferred)",
    )
    ap.add_argument("--output_dir", type=str, default="outputs/eval_runs/default")
    ap.add_argument("--config", type=str, default="model", help="Fallback model YAML if checkpoint lacks `model_cfg`")
    ap.add_argument("--device", type=str, default="cuda", help="cuda (default), cpu, or e.g. cuda:1")
    ap.add_argument("--batch_size", type=int, default=128, help="Test batch size")
    args = ap.parse_args()

    ckpt_path = _resolve(args.checkpoint)
    out_dir = _resolve(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(log_file=out_dir / "eval.log")
    device = resolve_compute_device(args.device)
    logger.info("Compute device: %s", device)

    ckpt = load_checkpoint(ckpt_path, map_location=device)

    if args.artifacts_dir:
        art_dir = _resolve(args.artifacts_dir)
    else:
        art_dir = infer_artifacts_dir_from_checkpoint(ckpt_path)
        if art_dir is None:
            for fb in (_resolve("outputs 2/artifacts"), _resolve("outputs/artifacts")):
                if (fb / "label_map.json").is_file():
                    art_dir = fb
                    logger.warning("Checkpoint path did not contain artifacts/; using %s", art_dir)
                    break

    if art_dir is None or not art_dir.exists():
        raise FileNotFoundError(
            f"Artifacts directory not found (tried inferring from checkpoint and common fallbacks). "
            f"Pass --artifacts_dir explicitly. Checkpoint was: {ckpt_path}"
        )

    out_root = art_dir.parent
    cache_dir = out_root / "cache" / "wisdm_windows"
    if not cache_dir.exists():
        raise FileNotFoundError(f"Missing window cache directory: {cache_dir}. Run preprocessing first.")

    splits = load_splits(art_dir)
    train_ids, val_ids, test_ids = subject_split_ids(splits)
    assert_disjoint_subject_sets(train_ids, val_ids, test_ids)

    label_map = load_label_map(art_dir)
    num_classes = int(ckpt.get("num_classes", label_map["num_classes"]))
    index_to_code: list[str] = list(label_map["index_to_code"])
    class_names = [f"{i}:{index_to_code[i]}" for i in range(num_classes)]
    labels = list(range(num_classes))

    norm = load_norm_stats(art_dir)
    yaml_cfg = load_merged_config(args.config)
    probe_cfg = probe_cfg_from_checkpoint(ckpt, yaml_cfg)
    test_ds = WISDMSupervisedDataset(
        cache_dir,
        test_ids,
        mean=norm.mean,
        std=norm.std,
        feature_stack=probe_cfg["feature_stack"],
    )
    bs = max(1, int(args.batch_size))
    loader = DataLoader(
        test_ds,
        batch_size=bs,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    model_cfg = merge_snn_model_cfg_from_checkpoint(ckpt.get("model_cfg"), yaml_cfg)
    if ckpt.get("model_cfg") is None:
        logger.warning("Checkpoint missing `model_cfg`; merged defaults from `--config %s`.", args.config)

    backbone = build_snn_backbone(model_cfg).to(device)
    head = linear_probe_head_from_cfg(backbone.out_dim, num_classes, probe_cfg).to(device)

    backbone.load_state_dict(ckpt["backbone"], strict=True)
    head.load_state_dict(ckpt["head"], strict=True)

    freeze_module(backbone)
    assert_backbone_frozen(backbone)

    backbone.eval()
    head.eval()

    ys: list[int] = []
    ps: list[int] = []
    with torch.no_grad():
        for x, y, _w in loader:
            x = x.to(device)
            logits = head(backbone(to_bct(x)))
            pred = torch.argmax(logits, dim=-1).detach().cpu().numpy().tolist()
            ys.extend(y.detach().cpu().numpy().tolist())
            ps.extend(pred)

    y_true = np.asarray(ys, dtype=np.int64)
    y_pred = np.asarray(ps, dtype=np.int64)

    metrics = compute_metrics(y_true, y_pred, labels=labels)
    metrics["eval_checkpoint_path"] = str(ckpt_path)
    metrics["artifacts_dir"] = str(art_dir.resolve())
    metrics["test_batch_size"] = bs
    write_json(out_dir / "metrics.json", metrics)

    report_txt = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=class_names,
        digits=4,
        zero_division=0,
    )
    (out_dir / "classification_report.txt").write_text(report_txt, encoding="utf-8")

    row = {
        "macro_f1": metrics["macro_f1"],
        "weighted_f1": metrics["weighted_f1"],
        "accuracy": metrics["accuracy"],
        "cohen_kappa": metrics["cohen_kappa"],
    }
    with (out_dir / "metrics_summary.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        w.writeheader()
        w.writerow(row)

    cm = np.asarray(metrics["confusion_matrix"], dtype=np.int64)
    save_confusion_matrix_figure(
        cm,
        class_names=class_names,
        out_path=out_dir / "confusion_matrix.png",
        title="Confusion matrix (test)",
    )

    logger.info("Wrote metrics to %s", out_dir / "metrics.json")


if __name__ == "__main__":
    main()
