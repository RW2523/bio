"""Time warping for AugPred-style SSL (pure PyTorch)."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def time_warp_stretch(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """
    Random temporal stretch/compress via 1D linear resampling.

    Parameters
    ----------
    x: [B, T, C]
    sigma: maximum relative stretch amount in (0, 0.5) recommended

    Returns
    -------
    x_warped: [B, T, C]
    """
    if x.ndim != 3:
        raise ValueError(f"Expected x [B,T,C], got {tuple(x.shape)}")
    b, t, c = x.shape
    if t < 4:
        return x
    sigma = float(sigma)
    if sigma <= 0:
        return x

    outs: list[torch.Tensor] = []
    for i in range(b):
        scale = float(torch.empty(1, device=x.device, dtype=x.dtype).uniform_(1.0 - sigma, 1.0 + sigma))
        t2 = max(2, int(round(t * scale)))
        y = F.interpolate(x[i : i + 1].permute(0, 2, 1), size=t2, mode="linear", align_corners=False)
        y = F.interpolate(y, size=t, mode="linear", align_corners=False).permute(0, 2, 1)
        outs.append(y)
    return torch.cat(outs, dim=0)
