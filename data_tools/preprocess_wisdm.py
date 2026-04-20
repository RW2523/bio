#!/usr/bin/env python3
"""Build window cache + split files + train-only normalization stats for WISDM raw streams."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_tools.cache_utils import ensure_dir
from data_tools.normalization import compute_channel_mean_std
from data_tools.parsers import infer_sample_rate_hz, load_activity_key, load_raw_timeseries, parse_raw_filename
from data_tools.splits import split_subjects
from data_tools.windowing import make_windows, map_activity_codes_to_indices
from utils.io import write_json
from utils.logger import setup_logger
from utils.paths import project_root
from utils.yaml_config import load_merged_config


def _resolve_path(p: str | Path) -> Path:
    pp = Path(p)
    if pp.is_absolute():
        return pp
    return (project_root() / pp).resolve()


def _list_raw_files(data_root: Path, device: str, modality: str) -> list[Path]:
    d = data_root / "raw" / device / modality
    if not d.exists():
        raise FileNotFoundError(
            f"Expected raw stream directory at `{d}`. "
            "Verify `data_root` points at the extracted `wisdm-dataset` folder."
        )
    files = sorted(d.glob("data_*_*.txt"))
    if not files:
        raise FileNotFoundError(f"No raw `data_*.txt` files found under {d}")
    return files


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="preprocess", help="Config name under configs/ (without .yaml)")
    args = ap.parse_args()

    cfg = load_merged_config(args.config)
    logger = setup_logger(log_file=_resolve_path(cfg["output_dir"]) / "logs" / "preprocess.log")

    data_root = _resolve_path(cfg["data_root"])
    out_root = _resolve_path(cfg["output_dir"])
    cache_dir = out_root / str(cfg["cache_subdir"])
    art_dir = out_root / str(cfg["artifacts_subdir"])
    ensure_dir(cache_dir)
    ensure_dir(art_dir)

    device = str(cfg["sensor_device"]).lower()
    modality = str(cfg["sensor_modality"]).lower()
    nominal_hz = float(cfg["nominal_sample_rate_hz"])
    window_sec = float(cfg["window_sec"])
    stride_sec = float(cfg["stride_sec"])
    drop_mixed = bool(cfg["drop_mixed_label_windows"])
    infer_hz = bool(cfg["infer_sample_rate"])
    hz_tol = float(cfg["sample_rate_tolerance"])

    window_samples = int(round(window_sec * nominal_hz))
    stride_samples = int(round(stride_sec * nominal_hz))
    if window_samples <= 0 or stride_samples <= 0:
        raise ValueError("window_samples/stride_samples computed to non-positive; check timing config.")

    activity_path = data_root / "activity_key.txt"
    code_to_name = load_activity_key(activity_path)
    codes_sorted = sorted(code_to_name.keys())
    code_to_index = {c: i for i, c in enumerate(codes_sorted)}
    num_classes = len(codes_sorted)

    raw_files = _list_raw_files(data_root, device=device, modality=modality)
    subjects: list[int] = []
    for fp in raw_files:
        sid, mod, dev = parse_raw_filename(fp)
        if mod != modality or dev != device:
            raise AssertionError(f"Filename/device mismatch for {fp}")
        subjects.append(sid)
    subjects = sorted(set(subjects))

    dbg = cfg.get("debug_subjects") or []
    if dbg:
        want = {int(x) for x in dbg}
        subjects = [s for s in subjects if s in want]
    max_subjects = cfg.get("max_subjects")
    if max_subjects is not None:
        subjects = subjects[: int(max_subjects)]

    if len(subjects) < 3:
        logger.warning("Fewer than 3 subjects after filtering; splits may fail. Consider removing debug filters.")

    train_s, val_s, test_s = split_subjects(
        subjects,
        train_frac=float(cfg["train_frac"]),
        val_frac=float(cfg["val_frac"]),
        seed=int(cfg["split_seed"]),
    )

    logger.info("Device/modality: %s/%s | subjects=%d | window=%ds stride=%ds (%d/%d samples @%.2fHz)", device, modality, len(subjects), window_sec, stride_sec, window_samples, stride_samples, nominal_hz)
    logger.info("Split counts train/val/test: %d/%d/%d", len(train_s), len(val_s), len(test_s))

    per_file_rows: list[dict] = []
    dropped_mixed_total = 0
    kept_total = 0

    for fp in raw_files:
        sid, _, _ = parse_raw_filename(fp)
        if sid not in set(subjects):
            continue

        ts, xyz, labels = load_raw_timeseries(fp, sort_by_time=True)
        row: dict = {"path": str(fp.relative_to(data_root)), "subject_id": sid, "n_rows": int(xyz.shape[0])}
        if infer_hz:
            hz_hat, diag = infer_sample_rate_hz(ts)
            rel = abs(hz_hat - nominal_hz) / max(nominal_hz, 1e-6)
            if rel > hz_tol:
                logger.warning(
                    "Inferred sample rate for %s is %.3f Hz (nominal %.3f Hz; rel_err=%.3f). Using nominal for window sizing.",
                    fp.name,
                    hz_hat,
                    nominal_hz,
                    rel,
                )
            row.update(
                {
                    "inferred_hz": hz_hat,
                    "median_dt_ns": diag["median_dt_ns"],
                    "large_gap_frac": diag["large_gap_frac"],
                }
            )
        per_file_rows.append(row)

        if xyz.shape[0] < window_samples:
            logger.warning("Skipping %s: only %d rows (< window_samples=%d)", fp.name, xyz.shape[0], window_samples)
            continue

        X, y_codes, motion, stats = make_windows(xyz, labels, window_samples, stride_samples, drop_mixed=drop_mixed)
        dropped_mixed_total += stats.dropped_mixed_windows
        kept_total += stats.kept_windows
        if X.shape[0] == 0:
            logger.warning("No windows extracted from %s", fp.name)
            continue

        y = map_activity_codes_to_indices(y_codes, code_to_index)

        out_npz = cache_dir / f"subject_{sid}.npz"
        np.savez_compressed(
            out_npz,
            X=X.astype(np.float32, copy=False),
            y=y.astype(np.int64, copy=False),
            motion=motion.astype(np.float32, copy=False),
            subject_id=np.int32(sid),
        )

    logger.info("Windowing summary: kept=%d dropped_mixed=%d", kept_total, dropped_mixed_total)

    # Train-only normalization stats
    train_X_parts: list[np.ndarray] = []
    for sid in train_s:
        p = cache_dir / f"subject_{sid}.npz"
        if not p.exists():
            continue
        z = np.load(p)
        train_X_parts.append(z["X"])

    if not train_X_parts:
        raise RuntimeError("No training windows found to compute normalization stats.")
    X_train = np.concatenate(train_X_parts, axis=0)
    mean, std = compute_channel_mean_std(X_train)
    norm_stats = {
        "mean": mean.tolist(),
        "std": std.tolist(),
        "computed_from": "train_windows_only",
        "n_windows": int(X_train.shape[0]),
    }
    write_json(art_dir / "norm_stats.json", norm_stats)

    label_artifact = {
        "num_classes": num_classes,
        "code_to_index": code_to_index,
        "code_to_name": code_to_name,
        "index_to_code": codes_sorted,
    }
    write_json(art_dir / "label_map.json", label_artifact)
    write_json(
        art_dir / "splits.json",
        {"train": train_s, "val": val_s, "test": test_s, "split_seed": int(cfg["split_seed"])},
    )
    write_json(
        art_dir / "preprocess_run.json",
        {
            "data_root": str(data_root),
            "sensor_device": device,
            "sensor_modality": modality,
            "window_sec": window_sec,
            "stride_sec": stride_sec,
            "nominal_hz": nominal_hz,
            "window_samples": window_samples,
            "stride_samples": stride_samples,
            "drop_mixed_label_windows": drop_mixed,
            "dropped_mixed_windows": dropped_mixed_total,
            "kept_windows": kept_total,
            "files": per_file_rows,
        },
    )

    write_json(
        art_dir / "windows_index.json",
        {
            "cache_dir": str(cache_dir.relative_to(out_root)),
            "files": [f"subject_{sid}.npz" for sid in subjects if (cache_dir / f"subject_{sid}.npz").exists()],
        },
    )

    logger.info("Wrote artifacts to %s and cache to %s", art_dir, cache_dir)


if __name__ == "__main__":
    main()
