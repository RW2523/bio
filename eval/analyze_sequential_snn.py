#!/usr/bin/env python3
"""
Lightweight explainability for the sequential SNN path: input–reservoir spike coupling.

Builds a coarse interaction matrix by correlating per-window TR spike counts (per input feature)
with reservoir neuron spike counts over the same window, on a sample of validation data.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.wisdm_supervised_dataset import WISDMSupervisedDataset
from models.encoders.tr_spike_encoder import TRSpikeEncoder
from models.sequential_snn_reservoir import ReservoirGeometry, SequentialSNNReservoir
from models.spike_neuron import lif_step
from models.stdp import stdp_config_from_mapping
from train.common import load_norm_stats, load_splits, resolve_compute_device, subject_split_ids
from train.train_wisdm_sequential_snn import _infer_cache_channels, _norm_for_channels, _slice_batch_x
from utils.checkpoint import load_checkpoint
from utils.io import write_json
from utils.paths import project_root


def _resolve(p: str) -> Path:
    pp = Path(p)
    return pp if pp.is_absolute() else (project_root() / pp).resolve()


@torch.no_grad()
def _reservoir_spike_counts(
    reservoir: SequentialSNNReservoir,
    spikes: torch.Tensor,
) -> torch.Tensor:
    """Return per-neuron spike counts ``[B, N]`` over time (no STDP)."""
    reservoir.set_stdp_enabled(False)
    b, t, f = spikes.shape
    device = spikes.device
    dtype = spikes.dtype
    state = reservoir.init_state(b, device, dtype)
    reservoir.reset_state(state)
    v = state["v"]
    w_rec_eff = reservoir.W_rec * reservoir.rec_mask.to(dtype=reservoir.W_rec.dtype)
    s_prev = torch.zeros(b, reservoir.n, device=device, dtype=dtype)
    counts = torch.zeros(b, reservoir.n, device=device, dtype=dtype)
    for tt in range(t):
        inp = spikes[:, tt] @ reservoir.W_in.T
        rec = s_prev @ w_rec_eff.T
        current = inp + rec
        v, spike = lif_step(
            v,
            current,
            beta=reservoir.lif_beta,
            threshold=reservoir.lif_threshold,
            reset_subtract=True,
        )
        counts = counts + spike
        s_prev = spike
    return counts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--artifacts_dir", type=str, default="")
    ap.add_argument("--output_dir", type=str, default="outputs/eval_runs/sequential_snn_analysis")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--max_batches", type=int, default=30)
    args = ap.parse_args()

    ckpt_path = _resolve(args.checkpoint)
    out_dir = _resolve(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_compute_device(args.device)
    ckpt = load_checkpoint(ckpt_path, map_location=device)

    if args.artifacts_dir:
        art_dir = _resolve(args.artifacts_dir)
    else:
        inferred = (ckpt_path.parents[2] / "artifacts").resolve()
        art_dir = inferred if inferred.exists() else _resolve("outputs/artifacts")
    out_root = art_dir.parent
    cache_dir = out_root / "cache" / "wisdm_windows"

    splits = load_splits(art_dir)
    _train_ids, val_ids, _test_ids = subject_split_ids(splits)
    meta = ckpt.get("meta") or {}
    use_gyro = bool(meta.get("use_gyro", False))
    base_norm = load_norm_stats(art_dir)
    cache_channels = _infer_cache_channels(cache_dir, val_ids)
    mean, std = _norm_for_channels(base_norm, cache_channels, use_gyro)
    ds = WISDMSupervisedDataset(cache_dir, val_ids, mean=mean, std=std)
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0, pin_memory=device.type == "cuda")

    enc_cfg = ckpt["encoder_cfg"]
    encoder = TRSpikeEncoder(
        num_channels=int(enc_cfg["num_channels"]),
        threshold=float(enc_cfg["threshold"]),
        threshold_scale=str(enc_cfg["threshold_scale"]),
        encoding=str(enc_cfg["encoding"]),
        delta_std=None,
    ).to(device)
    encoder.load_state_dict(ckpt["encoder_state"], strict=True)

    rc = ckpt["reservoir_cfg"]
    geom = ReservoirGeometry(nx=int(rc["nx"]), ny=int(rc["ny"]), nz=int(rc["nz"]))
    stdp_cfg = stdp_config_from_mapping(ckpt.get("stdp_cfg"))
    reservoir = SequentialSNNReservoir(
        in_features=int(rc["in_features"]),
        geometry=geom,
        lif_beta=float(rc["lif_beta"]),
        lif_threshold=float(rc["lif_threshold"]),
        cheb_radius=int(rc.get("cheb_radius", 1)),
        long_range_edges=int(rc.get("long_range_edges", 2)),
        weight_scale=float(rc.get("weight_scale", 0.35)),
        stdp_cfg=stdp_cfg,
        seed=int(rc.get("reservoir_seed", 42)),
    ).to(device)
    reservoir.load_state_dict(ckpt["reservoir"], strict=True)
    reservoir.eval()

    xs_in: list[np.ndarray] = []
    rs: list[np.ndarray] = []
    with torch.no_grad():
        for i, (x, _y, _w) in enumerate(loader):
            if i >= int(args.max_batches):
                break
            x = _slice_batch_x(x.to(device), cache_channels=cache_channels, use_gyro=use_gyro)
            sp = encoder(x)
            in_counts = sp.sum(dim=1).cpu().numpy()
            rcnts = _reservoir_spike_counts(reservoir, sp).cpu().numpy()
            xs_in.append(in_counts)
            rs.append(rcnts)

    Xin = np.concatenate(xs_in, axis=0)
    R = np.concatenate(rs, axis=0)
    Xin_c = Xin - Xin.mean(axis=0, keepdims=True)
    R_c = R - R.mean(axis=0, keepdims=True)
    num = (Xin_c.T @ R_c) / max(Xin.shape[0] - 1, 1)
    den = np.sqrt(np.sum(Xin_c**2, axis=0, keepdims=True).T * np.sum(R_c**2, axis=0, keepdims=True))
    corr = num / (den + 1e-8)

    per_channel = np.mean(np.abs(corr), axis=1).tolist()
    summary = {
        "interaction_shape": list(corr.shape),
        "mean_abs_corr": float(np.mean(np.abs(corr))),
        "per_input_feature_mean_abs_corr": per_channel,
    }
    write_json(out_dir / "interaction_summary.json", summary)

    fig, ax = plt.subplots(figsize=(12, 5))
    im = ax.imshow(corr, aspect="auto", cmap="magma", interpolation="nearest")
    ax.set_xlabel("reservoir neuron index")
    ax.set_ylabel("TR input feature index")
    ax.set_title("Correlation(TR spike counts, reservoir spike counts) per window")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_dir / "interaction_heatmap.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
