#!/usr/bin/env python3
"""Summarize cached windows: per-split counts, label histograms, subject counts."""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train.common import load_splits
from utils.io import write_json
from utils.paths import project_root


def _resolve(p: str) -> Path:
    pp = Path(p)
    return pp if pp.is_absolute() else (project_root() / pp).resolve()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts_dir", type=str, default="outputs/artifacts")
    args = ap.parse_args()

    art_dir = _resolve(args.artifacts_dir)
    out_root = art_dir.parent
    cache_dir = out_root / "cache" / "wisdm_windows"
    splits = load_splits(art_dir)
    split_lists = {k: v for k, v in splits.items() if isinstance(v, list)}

    summary: dict = {"subjects": {k: len(v) for k, v in split_lists.items()}, "windows": {}, "labels": {}}

    for split_name, sids in split_lists.items():
        y_all: list[int] = []
        nwin = 0
        for sid in sids:
            p = cache_dir / f"subject_{int(sid)}.npz"
            if not p.exists():
                continue
            z = np.load(p)
            y_all.extend(z["y"].astype(np.int64).tolist())
            nwin += int(z["y"].shape[0])
        summary["windows"][split_name] = int(nwin)
        summary["labels"][split_name] = dict(Counter(y_all))

    out = art_dir / "dataset_stats_summary.json"
    write_json(out, summary)


if __name__ == "__main__":
    main()
