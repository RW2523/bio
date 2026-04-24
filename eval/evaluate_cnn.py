#!/usr/bin/env python3
"""Evaluate a trained CNN linear-probe checkpoint on held-out test subjects."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.wisdm_supervised_dataset import WISDMSupervisedDataset
from eval.confusion_matrix import save_confusion_matrix_figure
from eval.metrics import compute_metrics
from train.cnn_common import build_cnn_backbone, merge_cnn_model_cfg_from_checkpoint
from train.common import (
    freeze_module,
    linear_probe_head_from_cfg,
    load_label_map,
    load_norm_stats,
    load_splits,
    probe_cfg_from_checkpoint,
    resolve_compute_device,
    subject_split_ids,
    to_bct,
)
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
    ap.add_argument("--checkpoint", type=str, required=True, help="Path to CNN probe `best.pt`")
    ap.add_argument("--artifacts_dir", type=str, default="")
    ap.add_argument("--output_dir", type=str, default="outputs/eval_runs/cnn_default")
    ap.add_argument("--config", type=str, default="cnn_model", help="Fallback CNN model YAML if checkpoint lacks `model_cfg`")
    ap.add_argument("--device", type=str, default="cuda", help="cuda, cpu, or e.g. cuda:1")
    args = ap.parse_args()

    ckpt_path = _resolve(args.checkpoint)
    out_dir = _resolve(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(log_file=out_dir / "eval.log")
    device = resolve_compute_device(args.device)
    logger.info("Compute device: %s", device)

    ckpt = load_checkpoint(ckpt_path, map_location=device)
    if ckpt.get("backbone_type") not in {None, "cnn_resnet1d"}:
        raise ValueError(f"Expected CNN probe checkpoint, got backbone_type={ckpt.get('backbone_type')!r}")

    if args.artifacts_dir:
        art_dir = _resolve(args.artifacts_dir)
    else:
        inferred = (ckpt_path.parents[2] / "artifacts").resolve()
        art_dir = inferred if inferred.exists() else _resolve("outputs/artifacts")
    if not art_dir.exists():
        raise FileNotFoundError(f"Artifacts directory not found: {art_dir}")

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
    model_cfg = merge_cnn_model_cfg_from_checkpoint(ckpt.get("model_cfg"), yaml_cfg)
    if ckpt.get("model_cfg") is None:
        logger.warning("Checkpoint missing `model_cfg`; merged CNN architecture from `--config %s`.", args.config)
    probe_yaml = probe_cfg_from_checkpoint(ckpt, yaml_cfg)

    test_ds = WISDMSupervisedDataset(
        cache_dir, test_ids, mean=norm.mean, std=norm.std, feature_stack=probe_yaml["feature_stack"]
    )
    loader = DataLoader(test_ds, batch_size=128, shuffle=False, num_workers=0, pin_memory=device.type == "cuda")

    backbone = build_cnn_backbone(model_cfg).to(device)
    head = linear_probe_head_from_cfg(backbone.out_dim, num_classes, probe_yaml).to(device)
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
            pred = torch.argmax(head(backbone(to_bct(x.to(device)))), dim=-1)
            ys.extend(y.detach().cpu().numpy().tolist())
            ps.extend(pred.detach().cpu().numpy().tolist())

    y_true = np.asarray(ys, dtype=np.int64)
    y_pred = np.asarray(ps, dtype=np.int64)
    metrics = compute_metrics(y_true, y_pred, labels=labels)
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

    save_confusion_matrix_figure(
        np.asarray(metrics["confusion_matrix"], dtype=np.int64),
        class_names=class_names,
        out_path=out_dir / "confusion_matrix.png",
        title="CNN confusion matrix (test)",
    )
    logger.info("Wrote metrics to %s", out_dir / "metrics.json")


if __name__ == "__main__":
    main()
