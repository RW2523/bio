"""Supervised window dataset (post-`preprocess_wisdm.py`)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from data_tools.normalization import normalize_windows

from datasets.window_features import normalize_feature_stack, stack_window_features


class WISDMSupervisedDataset(Dataset):
    """
    Loads cached per-subject `.npz` windows produced by preprocessing.

    Each item is `(x, y, sample_weight)` where `x` is float32 `[T, C]` normalized with train stats.
    `sample_weight` is derived from motion magnitude (window mean per-channel std) for optional sampling.
    """

    def __init__(
        self,
        cache_dir: Path,
        subject_ids: list[int],
        mean: np.ndarray,
        std: np.ndarray,
        *,
        feature_stack: list[str] | str | None = None,
    ) -> None:
        super().__init__()
        self.cache_dir = Path(cache_dir)
        self.mean = mean.astype(np.float32, copy=False)
        self.std = std.astype(np.float32, copy=False)
        self.feature_stack: list[str] = normalize_feature_stack(feature_stack)

        parts: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        expected_c = int(mean.shape[0])
        t_ref: int | None = None
        for sid in subject_ids:
            p = self.cache_dir / f"subject_{sid}.npz"
            if not p.exists():
                raise FileNotFoundError(f"Missing cached windows for subject {sid}: {p}")
            z = np.load(p)
            for key in ("X", "y", "motion"):
                if key not in z:
                    raise KeyError(f"{p} missing key {key!r}")
            X, y, motion = z["X"], z["y"], z["motion"]
            if X.ndim != 3 or X.shape[2] != expected_c:
                raise ValueError(f"{p}: expected X [N,T,{expected_c}], got {X.shape}")
            if t_ref is None:
                t_ref = int(X.shape[1])
            elif int(X.shape[1]) != t_ref:
                raise ValueError(f"{p}: time length {X.shape[1]} != reference {t_ref}")
            if y.shape[0] != X.shape[0] or motion.shape[0] != X.shape[0]:
                raise ValueError(f"{p}: length mismatch X={X.shape[0]} y={y.shape[0]} motion={motion.shape[0]}")
            parts.append((X, y, motion))

        lengths = np.array([p[0].shape[0] for p in parts], dtype=np.int64)
        self.cumlen = np.concatenate([[0], np.cumsum(lengths)])
        self.parts = parts
        self.total = int(self.cumlen[-1])

        motions_list = [p[2] for p in parts]
        self._motions = np.concatenate(motions_list, axis=0).astype(np.float64, copy=False)

    def count_labels(self, num_classes: int) -> np.ndarray:
        """Per-class window counts over all concatenated subjects (for class-balanced loss)."""
        ys = [np.asarray(y, dtype=np.int64).reshape(-1) for _, y, _ in self.parts]
        all_y = np.concatenate(ys, axis=0) if ys else np.zeros((0,), dtype=np.int64)
        return np.bincount(all_y, minlength=num_classes).astype(np.int64)

    @property
    def motions(self) -> np.ndarray:
        return self._motions

    def __len__(self) -> int:
        return self.total

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if idx < 0 or idx >= self.total:
            raise IndexError(idx)
        part_idx = int(np.searchsorted(self.cumlen, idx, side="right") - 1)
        local = int(idx - self.cumlen[part_idx])
        X, y, motion = self.parts[part_idx]
        x = normalize_windows(X[local : local + 1], self.mean, self.std)[0]
        x = stack_window_features(x, self.feature_stack)
        x_t = torch.from_numpy(np.ascontiguousarray(x))  # [T, C_out]
        y_t = torch.tensor(int(y[local]), dtype=torch.long)
        w_t = torch.tensor(float(motion[local]), dtype=torch.float32)
        return x_t, y_t, w_t
