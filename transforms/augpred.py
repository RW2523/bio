"""AugPred-style augmentation utilities and targets."""

from __future__ import annotations

import torch

from transforms.timewarp import time_warp_stretch


def chunk_shuffle_time(x: torch.Tensor, num_chunks: int) -> torch.Tensor:
    """
    Shuffle contiguous time chunks (different from identity with high probability).

    x: [B, T, C]
    """
    if x.ndim != 3:
        raise ValueError(f"Expected x [B,T,C], got {tuple(x.shape)}")
    b, t, c = x.shape
    k = int(num_chunks)
    if k <= 1 or t < k:
        return x

    chunk_size = int(torch.div(t, k, rounding_mode="floor"))
    if chunk_size <= 0:
        return x

    usable = chunk_size * k
    tail = x[:, usable:, :]
    body = x[:, :usable, :].reshape(b, k, chunk_size, c)

    out_body = torch.empty_like(body)
    for i in range(b):
        perm = torch.randperm(k, device=x.device)
        # avoid accidental identity permutation for k>1
        if k > 1 and torch.equal(perm, torch.arange(k, device=x.device)):
            perm = (perm + 1) % k
        out_body[i] = body[i, perm]
    out = out_body.reshape(b, usable, c)
    if tail.numel() > 0:
        out = torch.cat([out, tail], dim=1)
    return out


def sample_arrow_of_time(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns augmented x and binary target in {0,1} float with shape [B] (1=reversed)."""
    b = x.shape[0]
    rev = torch.rand(b, device=x.device) < 0.5
    y = rev.to(dtype=torch.float32)
    out = x.clone()
    for i in range(b):
        if bool(rev[i].item()):
            out[i] = torch.flip(out[i], dims=[0])
    return out, y


def sample_permutation(x: torch.Tensor, num_chunks: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns augmented x and binary target (1=shuffled)."""
    b = x.shape[0]
    shuf = torch.rand(b, device=x.device) < 0.5
    y = shuf.to(dtype=torch.float32)
    out = x.clone()
    for i in range(b):
        if bool(shuf[i].item()):
            out[i] = chunk_shuffle_time(out[i : i + 1], num_chunks=num_chunks)[0]
    return out, y


def sample_time_warp(x: torch.Tensor, sigma: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns augmented x and binary target (1=warped)."""
    b = x.shape[0]
    warp = torch.rand(b, device=x.device) < 0.5
    y = warp.to(dtype=torch.float32)
    out = x.clone()
    for i in range(b):
        if bool(warp[i].item()):
            out[i] = time_warp_stretch(out[i : i + 1], sigma=sigma)[0]
    return out, y
