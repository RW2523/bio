"""Discrete-time leaky integrate-and-fire (LIF) helpers for reservoir neurons."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class DiscreteLIFParams:
    """Parameters for a simple discrete LIF used in the sequential reservoir."""

    beta: float = 0.9
    threshold: float = 1.0
    reset_subtract: bool = True


def lif_step(
    v: torch.Tensor,
    current: torch.Tensor,
    *,
    beta: float,
    threshold: float,
    reset_subtract: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    One discrete timestep: leaky integration then threshold.

    Args:
        v: membrane ``[B, N]``
        current: input current ``[B, N]``
        beta: leak factor in ``(0,1)``
        threshold: firing threshold (positive)
        reset_subtract: if True, subtract ``threshold`` on spike; else multiply membrane by ``1 - spike``

    Returns:
        ``(v_new, spike)`` with ``spike`` in ``{0,1}`` float ``[B, N]``.
    """
    v = beta * v + (1.0 - beta) * current
    spike = (v >= threshold).to(v.dtype)
    if reset_subtract:
        v = v - spike * threshold
    else:
        v = v * (1.0 - spike)
    return v, spike
