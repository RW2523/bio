"""Pair-based STDP with exponential traces (Song–Abbott style) for dense weight blocks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch


@dataclass
class STDPConfig:
    tau_plus: float = 20.0
    tau_minus: float = 20.0
    a_plus: float = 0.004
    a_minus: float = 0.003
    w_min: float = 0.0
    w_max: float = 1.0
    learning_rate: float = 0.01


def stdp_config_from_mapping(m: Mapping[str, Any] | None) -> STDPConfig:
    if not m:
        return STDPConfig()
    d = dict(m)
    return STDPConfig(
        tau_plus=float(d.get("tau_plus", 20.0)),
        tau_minus=float(d.get("tau_minus", 20.0)),
        a_plus=float(d.get("a_plus", 0.004)),
        a_minus=float(d.get("a_minus", 0.003)),
        w_min=float(d.get("w_min", 0.0)),
        w_max=float(d.get("w_max", 1.0)),
        learning_rate=float(d.get("learning_rate", 0.01)),
    )


def decay_traces(trace_pre: torch.Tensor, trace_post: torch.Tensor, *, tau_plus: float, tau_minus: float, dt: float = 1.0) -> None:
    """In-place exponential decay of traces (one timestep). Shapes ``[B, N]`` or ``[B, F]`` for trace_pre only in FF."""
    c_pre = float(torch.exp(torch.tensor(-dt / tau_plus, device=trace_pre.device, dtype=trace_pre.dtype)))
    c_post = float(torch.exp(torch.tensor(-dt / tau_minus, device=trace_post.device, dtype=trace_post.dtype)))
    trace_pre.mul_(c_pre)
    trace_post.mul_(c_post)


def stdp_update_recurrent(
    w: torch.Tensor,
    trace_pre: torch.Tensor,
    trace_post: torch.Tensor,
    pre_spike: torch.Tensor,
    post_spike: torch.Tensor,
    cfg: STDPConfig,
) -> None:
    """
    In-place STDP update for recurrent ``w[i, j]`` from presynaptic ``j`` to postsynaptic ``i``.

    Uses traces **after decay, before** incrementing with current spikes.

    On post spike at ``i``: ``Δw_ij += lr * A_plus * trace_pre[j]``.
    On pre spike at ``j``: ``Δw_ij -= lr * A_minus * trace_post[i]``.

    Shapes: ``w`` ``[N, N]``; spikes and traces ``[B, N]``; batch summed into one update.
    """
    if w.ndim != 2 or w.shape[0] != w.shape[1]:
        raise ValueError("stdp_update_recurrent expects square w [N,N]")
    lr = float(cfg.learning_rate)
    dw = torch.zeros_like(w)
    if post_spike.any():
        dw = dw + cfg.a_plus * lr * torch.einsum("bi,bj->ij", post_spike, trace_pre)
    if pre_spike.any():
        dw = dw - cfg.a_minus * lr * torch.einsum("bi,bj->ij", trace_post, pre_spike)
    w.add_(dw)
    w.clamp_(cfg.w_min, cfg.w_max)


def stdp_update_ff(
    w: torch.Tensor,
    trace_pre_in: torch.Tensor,
    trace_post_res: torch.Tensor,
    input_spike: torch.Tensor,
    post_spike: torch.Tensor,
    cfg: STDPConfig,
) -> None:
    """
    Feedforward ``w[i, k]`` from input feature ``k`` to reservoir neuron ``i``.

    ``trace_pre_in``: ``[B, F]`` (input-side trace), ``trace_post_res``: ``[B, N]`` (reservoir trace).
    ``input_spike``: ``[B, F]``, ``post_spike``: ``[B, N]``.
    """
    lr = float(cfg.learning_rate)
    dw = torch.zeros_like(w)
    if post_spike.any():
        dw = dw + cfg.a_plus * lr * torch.einsum("bi,bf->if", post_spike, trace_pre_in)
    if input_spike.any():
        dw = dw - cfg.a_minus * lr * torch.einsum("bi,bf->if", trace_post_res, input_spike)
    w.add_(dw)
    w.clamp_(cfg.w_min, cfg.w_max)
