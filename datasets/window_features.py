"""Optional per-window feature stacking (numpy, post-normalization)."""

from __future__ import annotations

from typing import Sequence

import numpy as np

_VALID = frozenset({"raw", "delta", "window_std"})


def normalize_feature_stack(stack: str | Sequence[str] | None) -> list[str]:
    """Return a validated list of feature keys; default ``['raw']``."""
    if stack is None:
        return ["raw"]
    if isinstance(stack, str):
        keys = [stack.strip().lower()]
    else:
        keys = [str(s).strip().lower() for s in stack]
    for k in keys:
        if k not in _VALID:
            raise ValueError(f"Unknown feature_stack entry {k!r}; use one of {sorted(_VALID)}")
    if "raw" not in keys:
        raise ValueError("feature_stack must include 'raw'")
    return keys


def stack_window_features(x: np.ndarray, feature_stack: Sequence[str]) -> np.ndarray:
    """
    Stack features along the channel axis.

    Parameters
    ----------
    x:
        Normalized window ``[T, C]`` (typically ``C == 3`` accel).
    feature_stack:
        Ordered keys from ``normalize_feature_stack``.

    Returns
    -------
    Array ``[T, C_out]`` where ``C_out`` is ``C * len(feature_stack)`` when each
    branch uses ``C`` channels (raw, delta, and broadcast std use the same ``C``).
    """
    if x.ndim != 2:
        raise ValueError(f"Expected x [T, C], got {x.shape}")
    t, c = int(x.shape[0]), int(x.shape[1])
    parts: list[np.ndarray] = []
    for kind in feature_stack:
        if kind == "raw":
            parts.append(x.astype(np.float32, copy=False))
        elif kind == "delta":
            d = np.zeros((t, c), dtype=np.float32)
            if t > 1:
                d[1:] = x[1:].astype(np.float32, copy=False) - x[:-1].astype(np.float32, copy=False)
            parts.append(d)
        elif kind == "window_std":
            st = x.astype(np.float64, copy=False).std(axis=0, keepdims=True).astype(np.float32)
            parts.append(np.broadcast_to(st, (t, c)).copy())
    return np.concatenate(parts, axis=-1)


def infer_in_channels_from_feature_stack(feature_stack: Sequence[str], *, base_channels: int = 3) -> int:
    """Each stack entry contributes ``base_channels`` output channels (concatenated)."""
    return int(base_channels) * len(list(feature_stack))


def infer_input_channels(feature_stack: Sequence[str], base_channels: int = 3) -> int:
    """Alias for YAML docs / configs that phrase the args as (stack, base_sensor_channels)."""
    return infer_in_channels_from_feature_stack(feature_stack, base_channels=base_channels)
