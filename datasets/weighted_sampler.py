"""Build a PyTorch WeightedRandomSampler from per-window motion scores."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import WeightedRandomSampler


def motion_weights(motion: np.ndarray, power: float = 1.0, eps: float = 1e-3) -> torch.Tensor:
    w = np.power(motion.astype(np.float64) + eps, power)
    w = w / np.mean(w)
    return torch.tensor(w, dtype=torch.double)


def build_weighted_sampler(motions: np.ndarray, power: float, eps: float) -> WeightedRandomSampler:
    w = motion_weights(motions, power=power, eps=eps)
    return WeightedRandomSampler(weights=w, num_samples=int(w.numel()), replacement=True)
