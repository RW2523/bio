#!/usr/bin/env python3
"""Extract frozen backbone embeddings for cached windows."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.wisdm_supervised_dataset import WISDMSupervisedDataset
from train.common import freeze_module, load_norm_stats, load_splits, resolve_compute_device, to_bct
from train.snn_common import build_snn_backbone, merge_snn_model_cfg_from_checkpoint
from utils.assertions import assert_backbone_frozen
from utils.checkpoint import load_checkpoint
from utils.paths import project_root
from utils.yaml_config import load_merged_config


def _resolve(p: str) -> Path:
    pp = Path(p)
    return pp if pp.is_absolute() else (project_root() / pp).resolve()


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone_checkpoint", type=str, required=True, help="backbone_best.pt or a probe `best.pt`")
    ap.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    ap.add_argument("--artifacts_dir", type=str, default="outputs/artifacts")
    ap.add_argument("--output", type=str, default="outputs/features/embeddings.npz")
    ap.add_argument("--config", type=str, default="model")
    ap.add_argument("--device", type=str, default="cuda", help="cuda (default), cpu, or e.g. cuda:1")
    args = ap.parse_args()

    device = resolve_compute_device(args.device)
    art_dir = _resolve(args.artifacts_dir)
    out_root = art_dir.parent
    cache_dir = out_root / "cache" / "wisdm_windows"

    splits = load_splits(art_dir)
    ids = [int(x) for x in splits[args.split]]

    norm = load_norm_stats(art_dir)
    yaml_cfg = load_merged_config(args.config)
    ckpt = load_checkpoint(_resolve(args.backbone_checkpoint), map_location=device)
    model_cfg = merge_snn_model_cfg_from_checkpoint(ckpt.get("model_cfg"), yaml_cfg)

    ds = WISDMSupervisedDataset(
        cache_dir, ids, mean=norm.mean, std=norm.std, feature_stack=model_cfg["feature_stack"]
    )
    loader = DataLoader(ds, batch_size=128, shuffle=False, num_workers=0, pin_memory=device.type == "cuda")

    backbone = build_snn_backbone(model_cfg).to(device)
    if "state_dict" in ckpt:
        state = ckpt["state_dict"]
    elif "backbone" in ckpt:
        state = ckpt["backbone"]
    else:
        raise KeyError("Checkpoint must contain `state_dict` (SSL) or `backbone` (probe).")
    backbone.load_state_dict(state, strict=True)
    freeze_module(backbone)
    assert_backbone_frozen(backbone)
    backbone.eval()

    zs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    for x, y, _w in loader:
        z = backbone(to_bct(x.to(device))).detach().cpu().numpy()
        zs.append(z)
        ys.append(y.detach().cpu().numpy())
    Z = np.concatenate(zs, axis=0)
    Y = np.concatenate(ys, axis=0)

    out_path = _resolve(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, Z=Z.astype(np.float32), y=Y.astype(np.int64), split=np.array(args.split))


if __name__ == "__main__":
    main()
