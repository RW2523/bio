"""Compact 3D-grid reservoir with distance-biased connectivity and optional STDP (NeuCube-inspired)."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from models.spike_neuron import lif_step
from models.stdp import STDPConfig, decay_traces, stdp_update_ff, stdp_update_recurrent


@dataclass(frozen=True)
class ReservoirGeometry:
    nx: int
    ny: int
    nz: int

    @property
    def n(self) -> int:
        return int(self.nx * self.ny * self.nz)


def _grid_positions(nx: int, ny: int, nz: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    xs = torch.arange(nx, device=device, dtype=dtype)
    ys = torch.arange(ny, device=device, dtype=dtype)
    zs = torch.arange(nz, device=device, dtype=dtype)
    gx, gy, gz = torch.meshgrid(xs, ys, zs, indexing="ij")
    return torch.stack([gx.reshape(-1), gy.reshape(-1), gz.reshape(-1)], dim=-1)


def build_small_world_recurrent_mask(
    pos: torch.Tensor,
    *,
    cheb_radius: int,
    long_range_edges: int,
    rng: torch.Generator | None,
) -> torch.Tensor:
    """
    Directed mask ``mask[i, j]=1`` allows connection **from j to i** (j presynaptic, i postsynaptic).

    Chebyshev neighbors within ``cheb_radius`` on the integer grid (excluding self),
    plus ``long_range_edges`` random targets per neuron.
    """
    n = pos.shape[0]
    device = pos.device
    mask = torch.zeros(n, n, dtype=torch.bool, device=device)
    diff = pos.unsqueeze(0) - pos.unsqueeze(1)
    cheb = diff.abs().max(dim=-1).values
    local = (cheb <= float(cheb_radius)) & (cheb > 0)
    mask |= local

    if long_range_edges > 0:
        for i in range(n):
            perm = torch.randperm(n, generator=rng, device=device)
            taken = 0
            for j in perm.tolist():
                if j == i:
                    continue
                mask[i, j] = True
                taken += 1
                if taken >= long_range_edges:
                    break
    mask.fill_diagonal_(False)
    return mask


class SequentialSNNReservoir(nn.Module):
    """
    Leaky integrate-and-fire reservoir on a 3D grid with sparse recurrent and feedforward weights.

    Consumes **encoded** spikes ``[B, T, F_in]`` and returns per-window summaries for classification.

    When ``stdp=True``, ``W_rec`` and ``W_in`` are updated in-place (unsupervised). When ``stdp=False``,
    dynamics only (frozen weights; no STDP).
    """

    def __init__(
        self,
        *,
        in_features: int,
        geometry: ReservoirGeometry,
        lif_beta: float = 0.9,
        lif_threshold: float = 1.0,
        cheb_radius: int = 1,
        long_range_edges: int = 2,
        weight_scale: float = 0.35,
        stdp_cfg: STDPConfig | None = None,
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.geom = geometry
        self.n = geometry.n
        self.in_features = int(in_features)
        self.lif_beta = float(lif_beta)
        self.lif_threshold = float(lif_threshold)
        self.stdp_cfg = stdp_cfg or STDPConfig()
        self._stdp_enabled = False

        g = torch.Generator(device="cpu")
        g.manual_seed(int(seed))

        pos = _grid_positions(geometry.nx, geometry.ny, geometry.nz, device=torch.device("cpu"), dtype=torch.float32)
        self.register_buffer("pos", pos.to(dtype=torch.float32))

        mask = build_small_world_recurrent_mask(
            pos,
            cheb_radius=int(cheb_radius),
            long_range_edges=int(long_range_edges),
            rng=g,
        )
        w_rec = torch.randn(self.n, self.n, generator=g) * float(weight_scale)
        w_rec = w_rec * mask.float()
        w_in = torch.randn(self.n, self.in_features, generator=g) * float(weight_scale)

        self.register_buffer("rec_mask", mask)
        self.W_rec = nn.Parameter(w_rec)
        self.W_in = nn.Parameter(w_in)

    def set_stdp_enabled(self, enabled: bool) -> None:
        self._stdp_enabled = bool(enabled)

    def init_state(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> dict[str, torch.Tensor]:
        b = int(batch_size)
        return {
            "v": torch.zeros(b, self.n, device=device, dtype=dtype),
            "trace_pre_r": torch.zeros(b, self.n, device=device, dtype=dtype),
            "trace_post_r": torch.zeros(b, self.n, device=device, dtype=dtype),
            "trace_pre_in": torch.zeros(b, self.in_features, device=device, dtype=dtype),
            "trace_post_in": torch.zeros(b, self.n, device=device, dtype=dtype),
        }

    @staticmethod
    def reset_state(state: dict[str, torch.Tensor]) -> None:
        for t in state.values():
            t.zero_()

    def forward(
        self,
        x_spikes: torch.Tensor,
        *,
        stdp: bool,
        state: dict[str, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Args:
            x_spikes: ``[B, T, F_in]`` (e.g. binary spikes).
            stdp: enable in-place STDP updates (guarded by ``set_stdp_enabled``).

        Returns:
            ``features`` ``[B, N]`` — mean firing rate per reservoir neuron over ``T``.
            ``state`` — last membrane / traces (for inspection).
        """
        if stdp and not self._stdp_enabled:
            raise RuntimeError("STDP requested but reservoir STDP is disabled (classifier / frozen stage).")

        if x_spikes.ndim != 3:
            raise ValueError(f"Expected x_spikes [B,T,F], got {tuple(x_spikes.shape)}")
        b, t, f = x_spikes.shape
        if f != self.in_features:
            raise ValueError(f"Expected F_in={self.in_features}, got {f}")

        device = x_spikes.device
        dtype = x_spikes.dtype
        if state is None:
            state = self.init_state(b, device, dtype)
        else:
            self.reset_state(state)

        v = state["v"]
        trace_pre_r = state["trace_pre_r"]
        trace_post_r = state["trace_post_r"]
        trace_pre_in = state["trace_pre_in"]
        trace_post_in = state["trace_post_in"]

        spike_sum = torch.zeros(b, self.n, device=device, dtype=dtype)
        w_rec_eff = self.W_rec * self.rec_mask.to(dtype=self.W_rec.dtype)

        cfg = self.stdp_cfg
        s_prev = torch.zeros(b, self.n, device=device, dtype=dtype)

        for tt in range(t):
            decay_traces(trace_pre_r, trace_post_r, tau_plus=cfg.tau_plus, tau_minus=cfg.tau_minus)
            decay_traces(trace_pre_in, trace_post_in, tau_plus=cfg.tau_plus, tau_minus=cfg.tau_minus)

            inp = x_spikes[:, tt] @ self.W_in.T
            rec = s_prev @ w_rec_eff.T
            current = inp + rec
            v, spike = lif_step(
                v,
                current,
                beta=self.lif_beta,
                threshold=self.lif_threshold,
                reset_subtract=True,
            )
            spike_sum = spike_sum + spike

            if stdp:
                stdp_update_recurrent(self.W_rec.data, trace_pre_r, trace_post_r, s_prev, spike, cfg)
                stdp_update_ff(self.W_in.data, trace_pre_in, trace_post_in, x_spikes[:, tt], spike, cfg)
                self.W_rec.data.mul_(self.rec_mask.to(self.W_rec.dtype))

            trace_pre_r = trace_pre_r + s_prev
            trace_post_r = trace_post_r + spike
            trace_pre_in = trace_pre_in + x_spikes[:, tt]
            trace_post_in = trace_post_in + spike

            s_prev = spike.detach()

        state["v"] = v
        state["trace_pre_r"] = trace_pre_r
        state["trace_post_r"] = trace_post_r
        state["trace_pre_in"] = trace_pre_in
        state["trace_post_in"] = trace_post_in

        features = spike_sum / max(t, 1)
        return features, state

