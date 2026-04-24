#!/usr/bin/env python3
"""Print preprocessing diagnostics from existing artifacts (no raw re-read)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train.common import load_label_map, load_splits, resolve_path, subject_split_ids
from utils.io import read_json
from utils.paths import project_root


def _resolve(p: str) -> Path:
    pp = Path(p)
    return pp if pp.is_absolute() else (project_root() / pp).resolve()


def _count_labels_for_split(cache_dir: Path, subject_ids: list[int], num_classes: int) -> np.ndarray:
    total = np.zeros(num_classes, dtype=np.int64)
    for sid in subject_ids:
        p = cache_dir / f"subject_{sid}.npz"
        if not p.exists():
            continue
        z = np.load(p)
        y = np.asarray(z["y"], dtype=np.int64).reshape(-1)
        total += np.bincount(y, minlength=num_classes)
    return total


def main() -> None:
    ap = argparse.ArgumentParser(description="Audit preprocess_run.json + per-split label counts.")
    ap.add_argument(
        "--artifacts_dir",
        type=str,
        required=True,
        help="Directory with preprocess_run.json, label_map.json, splits.json (e.g. outputs 2/artifacts)",
    )
    args = ap.parse_args()

    art_dir = _resolve(args.artifacts_dir)
    if not art_dir.is_dir():
        raise FileNotFoundError(str(art_dir))

    out_root = art_dir.parent
    cache_dir = out_root / "cache" / "wisdm_windows"
    if not cache_dir.is_dir():
        raise FileNotFoundError(f"Expected window cache at {cache_dir}")

    pre = read_json(art_dir / "preprocess_run.json")
    nominal = float(pre.get("nominal_hz", pre.get("nominal_sample_rate_hz", 0.0)))
    kept = int(pre.get("kept_windows", 0))
    dropped_mixed = int(pre.get("dropped_mixed_windows", 0))

    print("=== preprocess_run.json ===")
    print(f"nominal_hz: {nominal}")
    print(f"kept_windows: {kept}")
    print(f"dropped_mixed_label_windows (total): {dropped_mixed}")
    if kept + dropped_mixed > 0:
        print(f"fraction windows dropped (mixed / all): {dropped_mixed / (kept + dropped_mixed):.4f}")

    files = pre.get("files") or []
    if files:
        print("\n=== per-file inferred_hz vs nominal (subset) ===")
        rows = []
        for row in files:
            if "inferred_hz" in row:
                hz_hat = float(row["inferred_hz"])
                rel = abs(hz_hat - nominal) / max(nominal, 1e-6)
                rows.append((row.get("subject_id"), hz_hat, rel))
        if not rows:
            print("(no inferred_hz entries; infer_sample_rate may be false in preprocess config)")
        else:
            for sid, hz_hat, rel in sorted(rows, key=lambda t: t[2], reverse=True)[:20]:
                print(f"  subject={sid} inferred_hz={hz_hat:.4f} rel_err_vs_nominal={rel:.4f}")
            hz_all = [r[1] for r in rows]
            print(f"inferred_hz min/median/max: {min(hz_all):.4f} / {float(np.median(hz_all)):.4f} / {max(hz_all):.4f}")

    label_map = load_label_map(art_dir)
    num_classes = int(label_map["num_classes"])
    splits = load_splits(art_dir)
    train_ids, val_ids, test_ids = subject_split_ids(splits)

    print("\n=== class counts (windows) per split ===")
    for name, ids in ("train", train_ids), ("val", val_ids), ("test", test_ids):
        c = _count_labels_for_split(cache_dir, ids, num_classes)
        print(f"  {name}: n_windows={int(c.sum())} per_class={c.tolist()}")

    print("\n=== interpretation hints ===")
    print("- Large inferred vs nominal drift can mis-align window length in samples if nominal is wrong.")
    print("- High mixed-label drops shrink effective data and can remove rare transition classes.")
    print("- Severe class imbalance in train windows limits what CE / linear probes can learn.")


if __name__ == "__main__":
    main()
