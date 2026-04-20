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
from models.linear_probe import LinearProbeHead
from models.spiking_resnet1d import SpikingResNet1d
from sklearn.metrics import classification_report

from train.common import (
    freeze_module,
    load_label_map,
    load_norm_stats,
    load_splits,
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
    ap.add_argument("--checkpoint", type=str, required=True, help="Path to `best.pt` from case1/case2 training")
    ap.add_argument(
        "--artifacts_dir",
        type=str,
        default="",
        help="Directory containing label_map.json/norm_stats.json/splits.json/preprocess_run.json (default: inferred)",
    )
    ap.add_argument("--output_dir", type=str, default="outputs/eval_runs/default")
    ap.add_argument("--config", type=str, default="model", help="Fallback model YAML if checkpoint lacks `model_cfg`")
    ap.add_argument("--device", type=str, default="cuda", help="cuda (default), cpu, or e.g. cuda:1")
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
        inferred = (ckpt_path.parents[2] / "artifacts").resolve()  # outputs/artifacts for outputs/checkpoints/<run>/best.pt
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
    test_ds = WISDMSupervisedDataset(cache_dir, test_ids, mean=norm.mean, std=norm.std)
    loader = DataLoader(
        test_ds,
        batch_size=128,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    model_cfg = ckpt.get("model_cfg")
    if model_cfg is None:
        cfg = load_merged_config(args.config)
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
        logger.warning("Checkpoint missing `model_cfg`; rebuilt architecture from `--config %s`.", args.config)

    backbone = SpikingResNet1d(
        in_channels=int(model_cfg["in_channels"]),
        base_channels=int(model_cfg["base_channels"]),
        layers=tuple(int(x) for x in model_cfg["layers"]),
        stem_kernel=int(model_cfg["stem_kernel"]),
        beta=float(model_cfg["lif_beta"]),
        threshold=float(model_cfg["lif_threshold"]),
        surrogate_alpha=float(model_cfg["surrogate_alpha"]),
        reset=str(model_cfg["lif_reset"]),
    ).to(device)
    head = LinearProbeHead(backbone.out_dim, num_classes=num_classes).to(device)

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
