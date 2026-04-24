#!/usr/bin/env python3
"""Print nominal vs inferred Hz per file from outputs/artifacts/preprocess_run.json."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.io import read_json
from utils.paths import project_root


def _resolve(p: str) -> Path:
    pp = Path(p)
    return pp if pp.is_absolute() else (project_root() / pp).resolve()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--artifacts_dir",
        type=str,
        default="outputs/artifacts",
        help="Directory containing preprocess_run.json",
    )
    args = ap.parse_args()
    art = _resolve(args.artifacts_dir)
    path = art / "preprocess_run.json"
    if not path.is_file():
        print(f"Missing {path}", file=sys.stderr)
        sys.exit(1)
    doc = read_json(path)
    nominal = float(doc.get("nominal_hz", float("nan")))
    print(f"nominal_sample_rate_hz (preprocess): {nominal}")
    print("path\tinferred_hz\tnominal_hz\tdelta_hz\tmedian_dt_ns")
    for row in doc.get("files", []):
        rel = row.get("path", "")
        inf = row.get("inferred_hz")
        med = row.get("median_dt_ns", "")
        if inf is None:
            print(f"{rel}\t(n/a)\t{nominal}\t(n/a)\t{med}")
        else:
            inf = float(inf)
            print(f"{rel}\t{inf:.6f}\t{nominal}\t{inf - nominal:.6f}\t{med}")


if __name__ == "__main__":
    main()
