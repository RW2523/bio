"""Subject-wise dataset splits."""

from __future__ import annotations

import numpy as np


def split_subjects(
    subject_ids: list[int],
    train_frac: float,
    val_frac: float,
    seed: int,
) -> tuple[list[int], list[int], list[int]]:
    if not (0 < train_frac < 1) or not (0 < val_frac < 1) or train_frac + val_frac >= 1:
        raise ValueError("Invalid train/val fractions; require train+val < 1 and all positive.")
    ids = sorted(set(int(s) for s in subject_ids))
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)
    n = len(ids)
    if n < 3:
        raise ValueError(f"Need at least 3 subjects for train/val/test splits; got {n}")

    n_train = int(round(train_frac * n))
    n_val = int(round(val_frac * n))
    n_train = max(1, min(n_train, n - 2))
    n_val = max(1, min(n_val, n - n_train - 1))
    n_test = n - n_train - n_val
    if n_test <= 0:
        raise ValueError("Empty test split after rounding; adjust fractions or subject count.")

    train = ids[:n_train]
    val = ids[n_train : n_train + n_val]
    test = ids[n_train + n_val :]
    return train, val, test
