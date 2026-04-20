"""Fixed-size sliding windows over labeled sensor streams."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class WindowingStats:
    total_windows: int
    kept_windows: int
    dropped_mixed_windows: int


def make_windows(
    xyz: np.ndarray,
    labels: np.ndarray,
    window_samples: int,
    stride_samples: int,
    drop_mixed: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, WindowingStats]:
    """
    Create sliding windows along time axis.

    Parameters
    ----------
    xyz: float array [N, 3]
    labels: length N single-character activity codes (numpy str_)
    window_samples / stride_samples: ints

    Returns
    -------
    X: [W, window_samples, 3]
    y_code: [W] single-character activity codes (numpy str_)
    motion: [W] float motion score (mean per-channel std over window)
    stats
    """
    n = int(xyz.shape[0])
    if labels.shape[0] != n:
        raise ValueError("labels length must match xyz rows")
    if window_samples <= 0 or stride_samples <= 0:
        raise ValueError("window_samples and stride_samples must be positive")

    xs: list[np.ndarray] = []
    y_codes: list[str] = []
    motions: list[float] = []

    dropped = 0
    total = 0
    for start in range(0, n - window_samples + 1, stride_samples):
        total += 1
        sl = slice(start, start + window_samples)
        win_labels = labels[sl]
        uniq = np.unique(win_labels)
        if drop_mixed and uniq.size != 1:
            dropped += 1
            continue
        if uniq.size != 1:
            # Non-default path: label window by first sample (ambiguous)
            code = str(win_labels[0])
        else:
            code = str(win_labels[0])
        w = xyz[sl].astype(np.float32, copy=False)
        motion = float(np.mean(np.std(w, axis=0)))
        xs.append(w)
        y_codes.append(code)
        motions.append(motion)

    if not xs:
        empty = np.zeros((0, window_samples, 3), dtype=np.float32)
        return empty, np.zeros((0,), dtype=np.str_), np.zeros((0,), dtype=np.float32), WindowingStats(total, 0, dropped)

    X = np.stack(xs, axis=0)
    motion_arr = np.asarray(motions, dtype=np.float32)
    y_arr = np.array(y_codes, dtype=np.str_)
    stats = WindowingStats(total_windows=total, kept_windows=X.shape[0], dropped_mixed_windows=dropped)
    return X, y_arr, motion_arr, stats


def map_activity_codes_to_indices(y_codes: np.ndarray, mapping: dict[str, int]) -> np.ndarray:
    out = np.empty((y_codes.shape[0],), dtype=np.int64)
    for i, c in enumerate(y_codes.tolist()):
        key = str(c)
        if key not in mapping:
            raise KeyError(f"Unknown activity code {key!r}")
        out[i] = int(mapping[key])
    return out
