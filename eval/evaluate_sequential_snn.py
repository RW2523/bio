#!/usr/bin/env python3
"""Evaluate a sequential SNN WISDM checkpoint on held-out test subjects."""

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
from models.desnn_classifier import build_output_head
from models.encoders.tr_spike_encoder import TRSpikeEncoder
from models.sequential_snn_reservoir import ReservoirGeometry, SequentialSNNReservoir
from models.stdp import stdp_config_from_mapping
from train.common import load_label_map, load_norm_stats, load_splits, resolve_compute_device, subject_split_ids
from train.train_wisdm_sequential_snn import _infer_cache_channels, _norm_for_channels, _slice_batch_x
from utils.assertions import assert_disjoint_subject_sets
from utils.checkpoint import load_checkpoint
from utils.io import write_json
from utils.logger import setup_logger
from utils.paths import project_root


def _resolve(p: str) -> Path:
    pp = Path(p)
    return pp if pp.is_absolute() else (project_root() / pp).resolve()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--artifacts_dir", type=str, default="", help="Default: ../../artifacts from checkpoint path")
    ap.add_argument("--output_dir", type=str, default="outputs/eval_runs/sequential_snn_test")
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()

    ckpt_path = _resolve(args.checkpoint)
    out_dir = _resolve(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(log_file=out_dir / "eval_sequential_snn.log")
    device = resolve_compute_device(args.device)
    logger.info("Compute device: %s", device)

    ckpt = load_checkpoint(ckpt_path, map_location=device)
    if ckpt.get("kind") != "sequential_snn_wisdm":
        logger.warning("Checkpoint kind is not 'sequential_snn_wisdm'; proceeding anyway.")

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
        raise FileNotFoundError(f"Missing window cache: {cache_dir}")

    splits = load_splits(art_dir)
    train_ids, val_ids, test_ids = subject_split_ids(splits)
    assert_disjoint_subject_sets(train_ids, val_ids, test_ids)

    label_map = load_label_map(art_dir)
    num_classes = int(ckpt.get("num_classes", label_map["num_classes"]))
    index_to_code: list[str] = list(label_map["index_to_code"])
    class_names = [f"{i}:{index_to_code[i]}" for i in range(num_classes)]
    labels = list(range(num_classes))

    meta = ckpt.get("meta") or {}
    use_gyro = bool(meta.get("use_gyro", False))
    base_norm = load_norm_stats(art_dir)
    cache_channels = _infer_cache_channels(cache_dir, test_ids)
    mean, std = _norm_for_channels(base_norm, cache_channels, use_gyro)

    test_ds = WISDMSupervisedDataset(cache_dir, test_ids, mean=mean, std=std)
    loader = DataLoader(
        test_ds,
        batch_size=128,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    enc_cfg = ckpt["encoder_cfg"]
    encoder = TRSpikeEncoder(
        num_channels=int(enc_cfg["num_channels"]),
        threshold=float(enc_cfg["threshold"]),
        threshold_scale=str(enc_cfg["threshold_scale"]),
        encoding=str(enc_cfg["encoding"]),
        delta_std=None,
    ).to(device)
    encoder.load_state_dict(ckpt["encoder_state"], strict=True)

    rc = ckpt["reservoir_cfg"]
    geom = ReservoirGeometry(nx=int(rc["nx"]), ny=int(rc["ny"]), nz=int(rc["nz"]))
    stdp_cfg = stdp_config_from_mapping(ckpt.get("stdp_cfg"))
    reservoir = SequentialSNNReservoir(
        in_features=int(rc["in_features"]),
        geometry=geom,
        lif_beta=float(rc["lif_beta"]),
        lif_threshold=float(rc["lif_threshold"]),
        cheb_radius=int(rc.get("cheb_radius", 1)),
        long_range_edges=int(rc.get("long_range_edges", 2)),
        weight_scale=float(rc.get("weight_scale", 0.35)),
        stdp_cfg=stdp_cfg,
        seed=int(rc.get("reservoir_seed", 42)),
    ).to(device)
    reservoir.load_state_dict(ckpt["reservoir"], strict=True)
    reservoir.set_stdp_enabled(False)
    for p in reservoir.parameters():
        p.requires_grad_(False)
    reservoir.eval()

    head_mode = str(ckpt.get("classifier_mode", "desnn"))
    hc = dict(ckpt.get("head_cfg") or {})
    head = build_output_head(
        head_mode,
        reservoir_dim=geom.n,
        num_classes=num_classes,
        desnn_hidden=hc.get("desnn_hidden", 64),
        desnn_temperature=float(hc.get("desnn_temperature", 1.0)),
        desnn_dropout=float(hc.get("desnn_dropout", 0.0)),
        probe_embedding_batchnorm=bool(hc.get("probe_embedding_batchnorm", False)),
    ).to(device)
    head.load_state_dict(ckpt["head"], strict=True)
    head.eval()

    ys: list[int] = []
    ps: list[int] = []
    with torch.no_grad():
        for x, y, _w in loader:
            x = _slice_batch_x(x.to(device), cache_channels=cache_channels, use_gyro=use_gyro)
            spikes = encoder(x)
            feats, _ = reservoir(spikes, stdp=False)
            pred = torch.argmax(head(feats), dim=-1)
            ys.extend(y.numpy().tolist())
            ps.extend(pred.cpu().numpy().tolist())

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

    with (out_dir / "metrics_summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["accuracy", "macro_f1", "weighted_f1", "cohen_kappa"])
        w.writerow([metrics["accuracy"], metrics["macro_f1"], metrics["weighted_f1"], metrics["cohen_kappa"]])

    cm = np.asarray(metrics["confusion_matrix"], dtype=np.int64)
    save_confusion_matrix_figure(cm, class_names, out_dir / "confusion_matrix.png", title="Test confusion (sequential SNN)")
    logger.info("Wrote metrics to %s", out_dir)


if __name__ == "__main__":
    main()
