#!/usr/bin/env python3
"""Recursively audit the WISDM folder: structure, modalities, subjects, labels, timing hints."""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_tools.discovery import discover_wisdm
from data_tools.parsers import infer_sample_rate_hz, load_activity_key, load_raw_timeseries
from utils.io import write_json
from utils.logger import setup_logger
from utils.paths import project_root


def _resolve_path(p: str | Path) -> Path:
    pp = Path(p)
    if pp.is_absolute():
        return pp
    return (project_root() / pp).resolve()


def _peek_text_head(path: Path, n: int = 40) -> list[str]:
    lines: list[str] = []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for _ in range(n):
            line = f.readline()
            if not line:
                break
            lines.append(line.rstrip("\n"))
    return lines


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, default="../wisdm-dataset", help="Path to extracted `wisdm-dataset` directory")
    ap.add_argument("--out_dir", type=str, default="outputs/audit", help="Directory to write audit artifacts (relative to project/)")
    args = ap.parse_args()

    data_root = _resolve_path(args.data_root)
    out_dir = _resolve_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(log_file=out_dir / "inspect.log")

    summary = discover_wisdm(data_root)
    machine = summary.to_dict()

    # activity key
    act_path = data_root / "activity_key.txt"
    code_to_name: dict[str, str] = {}
    if act_path.exists():
        code_to_name = load_activity_key(act_path)
    machine["activity_codes"] = sorted(code_to_name.keys())
    machine["activity_code_to_name"] = code_to_name

    # subjects from raw filenames
    raw_root = data_root / "raw"
    subjects_by_stream: dict[str, list[int]] = {}
    sample_raw: Path | None = None
    if raw_root.exists():
        for device_dir in sorted([p for p in raw_root.iterdir() if p.is_dir()]):
            for mod_dir in sorted([p for p in device_dir.iterdir() if p.is_dir()]):
                key = f"raw/{device_dir.name}/{mod_dir.name}"
                sids: list[int] = []
                for fp in sorted(mod_dir.glob("data_*_*.txt")):
                    # data_<sid>_<mod>_<dev>.txt
                    stem = fp.stem
                    parts = stem.split("_")
                    if len(parts) >= 2 and parts[0] == "data":
                        sids.append(int(parts[1]))
                    if sample_raw is None and fp.name.endswith(".txt"):
                        sample_raw = fp
                subjects_by_stream[key] = sorted(set(sids))
        machine["subjects_by_raw_stream"] = {k: {"count": len(v), "min": min(v) if v else None, "max": max(v) if v else None} for k, v in subjects_by_stream.items()}

    timing_notes: dict[str, object] = {}
    label_counter: Counter[str] = Counter()
    if sample_raw is not None:
        ts, _, labels = load_raw_timeseries(sample_raw, sort_by_time=True)
        hz, diag = infer_sample_rate_hz(ts)
        timing_notes["sample_file"] = str(sample_raw.relative_to(data_root))
        timing_notes["inferred_hz_from_median_dt"] = hz
        timing_notes.update({f"dt_{k}": v for k, v in diag.items()})
        # label distribution on a subsample for speed
        m = min(50_000, labels.shape[0])
        label_counter.update([str(x) for x in labels[:m]])

    machine["timing_notes"] = timing_notes
    machine["label_counts_subsample_first_50k_or_all"] = dict(label_counter)

    # ARFF header peek (if present)
    arff_root = data_root / "arff_files"
    arff_peek: dict[str, object] = {}
    if arff_root.exists():
        found = next(arff_root.rglob("*.arff"), None)
        if found is not None:
            arff_peek["sample_file"] = str(found.relative_to(data_root))
            arff_peek["header_lines"] = _peek_text_head(found, 60)
    machine["arff_peek"] = arff_peek

    write_json(out_dir / "dataset_audit.json", machine)

    # Human-readable report
    report_path = out_dir / "DATASET_AUDIT_REPORT.txt"
    lines: list[str] = []
    lines.append("WISDM Dataset Audit (auto-generated)")
    lines.append(f"data_root: {data_root}")
    lines.append("")
    lines.append("Top-level folders:")
    for name in machine["folders"]:
        lines.append(f"  - {name}")
    lines.append("")
    lines.append("Raw text stream groups (counts):")
    for k, v in sorted(machine.get("raw_txt_by_group", {}).items()):
        lines.append(f"  - {k}: {v} files")
    lines.append("")
    lines.append("ARFF groups (counts):")
    for k, v in sorted(machine.get("arff_by_group", {}).items()):
        lines.append(f"  - {k}: {v} files")
    lines.append("")
    lines.append("Detected activity codes:")
    lines.append("  " + ", ".join(machine.get("activity_codes", [])))
    lines.append("")
    lines.append("Issues / ambiguities:")
    for issue in machine.get("issues", []):
        lines.append(f"  - {issue}")
    if not machine.get("issues"):
        lines.append("  - none flagged by discovery rules")
    lines.append("")
    lines.append("Timing notes (representative raw file):")
    for k in sorted(machine.get("timing_notes", {}).keys()):
        lines.append(f"  - {k}: {machine['timing_notes'][k]}")
    lines.append("")
    lines.append("Notes:")
    lines.append("  - This project v1 uses RAW comma-separated streams under `raw/{phone|watch}/{accel|gyro}/`.")
    lines.append("  - ARFF files are aggregated window features; they are detected for completeness but not used by default.")
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    logger.info("Wrote %s and %s", out_dir / "dataset_audit.json", report_path)


if __name__ == "__main__":
    main()
