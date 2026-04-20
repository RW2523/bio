"""Self-supervised window dataset (same windows as supervised; used for AugPred)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from data_tools.normalization import normalize_windows


class WISDMSSLDataset(Dataset):
    """Returns `(x, motion)` tensors for SSL training (x is normalized)."""

    def __init__(
        self,
        cache_dir: Path,
        subject_ids: list[int],
        mean: np.ndarray,
        std: np.ndarray,
    ) -> None:
        super().__init__()
        self.cache_dir = Path(cache_dir)
        self.mean = mean.astype(np.float32, copy=False)
        self.std = std.astype(np.float32, copy=False)

        parts: list[tuple[np.ndarray, np.ndarray]] = []
        expected_c = int(mean.shape[0])
        t_ref: int | None = None
        for sid in subject_ids:
            p = self.cache_dir / f"subject_{sid}.npz"
            if not p.exists():
                raise FileNotFoundError(f"Missing cached windows for subject {sid}: {p}")
            z = np.load(p)
            for key in ("X", "motion"):
                if key not in z:
                    raise KeyError(f"{p} missing key {key!r}")
            X, motion = z["X"], z["motion"]
            if X.ndim != 3 or X.shape[2] != expected_c:
                raise ValueError(f"{p}: expected X [N,T,{expected_c}], got {X.shape}")
            if t_ref is None:
                t_ref = int(X.shape[1])
            elif int(X.shape[1]) != t_ref:
                raise ValueError(f"{p}: time length {X.shape[1]} != reference {t_ref}")
            if motion.shape[0] != X.shape[0]:
                raise ValueError(f"{p}: motion length {motion.shape[0]} != N {X.shape[0]}")
            parts.append((X, motion))

        lengths = np.array([p[0].shape[0] for p in parts], dtype=np.int64)
        self.cumlen = np.concatenate([[0], np.cumsum(lengths)])
        self.parts = parts
        self.total = int(self.cumlen[-1])

        motions_list = [p[1] for p in parts]
        self._motions = np.concatenate(motions_list, axis=0).astype(np.float64, copy=False)

    @property
    def motions(self) -> np.ndarray:
        """Per-window motion scores aligned with dataset indices (for weighted sampling)."""
        return self._motions

    def __len__(self) -> int:
        return self.total

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        if idx < 0 or idx >= self.total:
            raise IndexError(idx)
        part_idx = int(np.searchsorted(self.cumlen, idx, side="right") - 1)
        local = int(idx - self.cumlen[part_idx])
        X, motion = self.parts[part_idx]
        x = normalize_windows(X[local : local + 1], self.mean, self.std)[0]
        x_t = torch.from_numpy(x)
        w_t = torch.tensor(float(motion[local]), dtype=torch.float32)
        return x_t, w_t
