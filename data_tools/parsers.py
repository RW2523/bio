"""WISDM parsers for raw `.txt` streams and auxiliary metadata files."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np


_RAW_NAME_RE = re.compile(r"^data_(\d+)_(accel|gyro)_(phone|watch)\.txt$", re.IGNORECASE)


@dataclass(frozen=True)
class RawSample:
    subject_id: int
    activity_code: str
    timestamp_ns: int
    x: float
    y: float
    z: float


def parse_raw_filename(path: Path) -> tuple[int, str, str]:
    m = _RAW_NAME_RE.match(path.name)
    if not m:
        raise ValueError(f"Unrecognized raw filename pattern: {path.name}")
    sid = int(m.group(1))
    modality = m.group(2).lower()
    device = m.group(3).lower()
    return sid, modality, device


def load_activity_key(path: Path) -> dict[str, str]:
    """
    Parse `activity_key.txt` lines like `walking = A` into code -> canonical name.
    Codes are single-letter strings (e.g. 'A').
    """
    if not path.exists():
        raise FileNotFoundError(f"Missing activity_key.txt at {path}")
    code_to_name: dict[str, str] = {}
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            left, right = [p.strip() for p in line.split("=", 1)]
            code = right
            # Some files may include spaces; keep single-letter code
            if len(code) != 1:
                raise ValueError(f"Unexpected activity code format in line: {line!r}")
            code_to_name[code] = left.replace(" ", "_")
    if not code_to_name:
        raise ValueError(f"No activity codes parsed from {path}")
    return code_to_name


def iter_raw_samples(path: Path) -> Iterator[RawSample]:
    """Stream-parse a WISDM raw `.txt` file."""
    sid, _, _ = parse_raw_filename(path)
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line_no, line in enumerate(f, start=1):
            s = line.strip()
            if not s:
                continue
            if s.endswith(";"):
                s = s[:-1]
            parts = s.split(",")
            if len(parts) != 6:
                raise ValueError(f"{path}:{line_no}: expected 6 comma fields, got {len(parts)}: {line!r}")
            try:
                file_sid = int(parts[0])
                act = parts[1].strip()
                ts = int(parts[2])
                x, y, z = float(parts[3]), float(parts[4]), float(parts[5])
            except ValueError as e:
                raise ValueError(f"{path}:{line_no}: bad numeric fields: {line!r}") from e
            if file_sid != sid:
                raise ValueError(f"{path}:{line_no}: subject id mismatch filename={sid} row={file_sid}")
            if len(act) != 1:
                raise ValueError(f"{path}:{line_no}: expected single-letter activity code, got {act!r}")
            yield RawSample(subject_id=sid, activity_code=act, timestamp_ns=ts, x=x, y=y, z=z)


def load_raw_timeseries(path: Path, sort_by_time: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load a raw file into:
    - ts: int64 [N] timestamps (nanoseconds in this dataset)
    - xyz: float32 [N, 3]
    - labels: bytes? use U1 strings - store as numpy bytes0 or int mapping outside

    Returns labels as length-N unicode strings array (object array of single chars).
    """
    samples = list(iter_raw_samples(path))
    if not samples:
        raise ValueError(f"Empty raw file: {path}")
    ts = np.array([s.timestamp_ns for s in samples], dtype=np.int64)
    xyz = np.array([[s.x, s.y, s.z] for s in samples], dtype=np.float32)
    labels = np.array([s.activity_code for s in samples], dtype=np.str_)

    if sort_by_time:
        order = np.argsort(ts, kind="mergesort")
        ts = ts[order]
        xyz = xyz[order]
        labels = labels[order]
    return ts, xyz, labels


def infer_sample_rate_hz(ts_ns: np.ndarray) -> tuple[float, dict[str, float]]:
    """
    Infer sample rate from timestamp deltas (after sorting).
    Returns (hz, diagnostics).
    """
    if ts_ns.size < 2:
        raise ValueError("Need at least 2 samples to infer sample rate")
    dt = np.diff(ts_ns.astype(np.float64))
    pos = dt[dt > 0]
    if pos.size == 0:
        raise ValueError("No positive timestamp deltas; cannot infer sample rate")
    med = float(np.median(pos))
    hz = 1e9 / med if med > 0 else float("nan")
    diag = {
        "median_dt_ns": med,
        "mean_dt_ns": float(np.mean(pos)),
        "p95_dt_ns": float(np.percentile(pos, 95)),
        "max_dt_ns": float(np.max(pos)),
        "large_gap_frac": float(np.mean(pos > 5 * med)),
    }
    return hz, diag
