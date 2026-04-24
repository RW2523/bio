"""Normalize temporal pooling options from YAML / merged model_cfg."""

from __future__ import annotations

from typing import Any, Mapping


def canonical_temporal_pooling(pooling: str | Any, pool_bins: int | Any) -> tuple[str, int]:
    """
    Return canonical ``(pooling, pool_bins)`` for backbones.

    Supported ``pooling``:
    - ``mean``: global average over time → ``[B, D]``.
    - ``adaptive`` + ``pool_bins`` > 1: ``AdaptiveAvgPool1d(pool_bins)`` → ``[B, D * pool_bins]``.
    - ``adaptive4``: shorthand for adaptive pooling with 4 bins (same as ``adaptive`` + ``pool_bins: 4``).
    - ``attention``: learned softmax weights over time → ``[B, D]`` (``pool_bins`` ignored).
    """
    p = str(pooling if pooling is not None else "mean").strip().lower()
    b = int(pool_bins) if pool_bins is not None else 1
    if p == "adaptive4":
        return "adaptive", 4
    if p not in {"mean", "adaptive", "attention"}:
        raise ValueError(
            f"pooling must be one of mean, adaptive, adaptive4, attention; got {pooling!r}"
        )
    if p == "adaptive":
        b = max(1, b)
    else:
        b = max(1, b)
    return p, b


def temporal_pooling_from_yaml(cfg: Mapping[str, Any]) -> tuple[str, int]:
    return canonical_temporal_pooling(cfg.get("pooling", "mean"), cfg.get("pool_bins", 1))
