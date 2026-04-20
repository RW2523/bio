"""Train-only normalization utilities for window tensors."""

from __future__ import annotations

import numpy as np


def compute_channel_mean_std(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    X: [N, T, C] unnormalized windows
    Returns mean[C], std[C] computed over N and T.
    """
    if X.ndim != 3:
        raise ValueError(f"Expected X [N,T,C], got shape {X.shape}")
    v = X.reshape(-1, X.shape[-1]).astype(np.float64, copy=False)
    mean = v.mean(axis=0)
    std = v.std(axis=0)
    std = np.maximum(std, 1e-6)
    return mean.astype(np.float32), std.astype(np.float32)


def normalize_windows(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((X - mean.reshape(1, 1, -1)) / std.reshape(1, 1, -1)).astype(np.float32, copy=False)
