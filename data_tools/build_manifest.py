#!/usr/bin/env python3
"""Build a machine-readable manifest of discovered WISDM raw streams (+ optional ARFF inventory)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_tools.parsers import infer_sample_rate_hz, load_raw_timeseries, parse_raw_filename
from utils.io import write_json
from utils.logger import setup_logger
from utils.paths import project_root


def _resolve_path(p: str | Path) -> Path:
    pp = Path(p)
    if pp.is_absolute():
        return pp
    return (project_root() / pp).resolve()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, default="../wisdm-dataset")
    ap.add_argument("--out_dir", type=str, default="outputs/audit")
    args = ap.parse_args()

    data_root = _resolve_path(args.data_root)
    out_dir = _resolve_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(log_file=out_dir / "manifest.log")

    raw_root = data_root / "raw"
    if not raw_root.exists():
        raise FileNotFoundError(f"Missing raw root: {raw_root}")

    records: list[dict] = []
    for fp in sorted(raw_root.rglob("data_*_*.txt")):
        rel = fp.relative_to(data_root).as_posix()
        sid, modality, device = parse_raw_filename(fp)
        rec: dict = {
            "relative_path": rel,
            "subject_id": sid,
            "device": device,
            "modality": modality,
            "kind": "raw_txt",
        }
        try:
            ts, xyz, _labels = load_raw_timeseries(fp, sort_by_time=True)
            hz, diag = infer_sample_rate_hz(ts)
            rec.update(
                {
                    "n_rows": int(xyz.shape[0]),
                    "inferred_hz": hz,
                    "median_dt_ns": diag["median_dt_ns"],
                    "large_gap_frac": diag["large_gap_frac"],
                }
            )
        except Exception as e:  # noqa: BLE001 - manifest should be best-effort per file
            rec["error"] = repr(e)
        records.append(rec)

    arff_records: list[dict] = []
    arff_root = data_root / "arff_files"
    if arff_root.exists():
        for fp in sorted(arff_root.rglob("*.arff")):
            rel = fp.relative_to(data_root).as_posix()
            # filename pattern mirrors raw: data_<sid>_<mod>_<dev>.arff
            arff_records.append({"relative_path": rel, "kind": "arff"})

    payload = {"data_root": str(data_root), "raw_records": records, "arff_inventory": arff_records}
    write_json(out_dir / "manifest.json", payload)
    logger.info("Wrote manifest with %d raw records and %d arff paths", len(records), len(arff_records))


if __name__ == "__main__":
    main()
