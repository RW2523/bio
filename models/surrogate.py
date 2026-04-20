"""Surrogate-gradient spike nonlinearity (pure PyTorch autograd.Function)."""

from __future__ import annotations

import torch


class FastSigmoidSurrogate(torch.autograd.Function):
    """Forward hard threshold; backward fast-sigmoid surrogate (SpikeGPT-style family)."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, alpha: float) -> torch.Tensor:  # type: ignore[override]
        ctx.save_for_backward(x)
        ctx.alpha = float(alpha)
        return (x > 0).to(x.dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):  # type: ignore[override]
        (x,) = ctx.saved_tensors
        alpha = ctx.alpha
        denom = 1.0 + alpha * torch.abs(x)
        grad_input = grad_output / (denom * denom) * alpha
        return grad_input, None


def spike_fn(x: torch.Tensor, alpha: float) -> torch.Tensor:
    return FastSigmoidSurrogate.apply(x, alpha)
