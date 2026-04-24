#!/usr/bin/env python3
"""Compare backbone embeddings (random init vs SSL) with silhouette + UMAP PNG."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from umap import UMAP

from datasets.wisdm_supervised_dataset import WISDMSupervisedDataset
from train.common import load_norm_stats, load_splits, resolve_compute_device, resolve_path, subject_split_ids, to_bct
from train.snn_common import build_snn_backbone, merge_snn_model_cfg_from_checkpoint
from utils.checkpoint import load_checkpoint
from utils.io import read_json
from utils.paths import project_root
from utils.yaml_config import load_merged_config


def _resolve(p: str) -> Path:
    pp = Path(p)
    return pp if pp.is_absolute() else (project_root() / pp).resolve()


@torch.no_grad()
def _embed_backbone(
    backbone: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_batches: int,
) -> tuple[np.ndarray, np.ndarray]:
    backbone.eval()
    zs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    for i, (x, y, _w) in enumerate(loader):
        if i >= max_batches:
            break
        x = x.to(device)
        z = backbone(to_bct(x)).detach().cpu().numpy().astype(np.float32)
        zs.append(z)
        ys.append(y.numpy().astype(np.int64))
    return np.concatenate(zs, axis=0), np.concatenate(ys, axis=0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts_dir", type=str, required=True)
    ap.add_argument("--config", type=str, default="model", help="YAML for SNN architecture if checkpoint lacks model_cfg")
    ap.add_argument("--random_backbone_ckpt", type=str, default="", help="Optional case1 `best.pt` to match probe arch (uses backbone only)")
    ap.add_argument("--ssl_backbone_ckpt", type=str, required=True, help="e.g. .../backbone_best.pt from SSL")
    ap.add_argument("--split", type=str, default="val", choices=("train", "val", "test"))
    ap.add_argument("--max_batches", type=int, default=50)
    ap.add_argument("--output_dir", type=str, required=True)
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()

    art_dir = _resolve(args.artifacts_dir)
    out_dir = _resolve(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_root = art_dir.parent
    cache_dir = out_root / "cache" / "wisdm_windows"
    device = resolve_compute_device(args.device)

    splits = load_splits(art_dir)
    train_ids, val_ids, test_ids = subject_split_ids(splits)
    ids = {"train": train_ids, "val": val_ids, "test": test_ids}[args.split]

    norm = load_norm_stats(art_dir)

    yaml_cfg = load_merged_config(args.config)
    ssl_path = _resolve(args.ssl_backbone_ckpt)
    ck_ssl = load_checkpoint(ssl_path, map_location=device)
    model_cfg_ssl = merge_snn_model_cfg_from_checkpoint(ck_ssl.get("model_cfg"), yaml_cfg)

    ds = WISDMSupervisedDataset(
        cache_dir, ids, mean=norm.mean, std=norm.std, feature_stack=model_cfg_ssl["feature_stack"]
    )
    loader = DataLoader(ds, batch_size=128, shuffle=True, num_workers=0, pin_memory=device.type == "cuda")

    def load_bb(path: Path) -> torch.nn.Module:
        ck = load_checkpoint(path, map_location=device)
        mc = merge_snn_model_cfg_from_checkpoint(ck.get("model_cfg"), yaml_cfg)
        bb = build_snn_backbone(mc).to(device)
        if "state_dict" in ck:
            bb.load_state_dict(ck["state_dict"], strict=True)
        elif "backbone" in ck:
            bb.load_state_dict(ck["backbone"], strict=True)
        else:
            raise ValueError(f"Checkpoint {path} needs `state_dict` (SSL) or `backbone` (probe) weights.")
        return bb

    ssl_bb = load_bb(ssl_path)

    if args.random_backbone_ckpt:
        rand_path = _resolve(args.random_backbone_ckpt)
        ck_rand = load_checkpoint(rand_path, map_location=device)
        model_cfg_rand = merge_snn_model_cfg_from_checkpoint(ck_rand.get("model_cfg"), yaml_cfg)
        if int(model_cfg_rand["in_channels"]) != int(model_cfg_ssl["in_channels"]):
            raise ValueError(
                f"Random backbone in_channels={model_cfg_rand['in_channels']} != SSL "
                f"in_channels={model_cfg_ssl['in_channels']} (check checkpoints and --config)."
            )
        rand_bb = load_bb(rand_path)
    else:
        rand_bb = build_snn_backbone(model_cfg_ssl).to(device)

    z_r, y_r = _embed_backbone(rand_bb, loader, device, args.max_batches)
    z_s, y_s = _embed_backbone(ssl_bb, loader, device, args.max_batches)
    # align rows if different batch tail
    n = min(len(z_r), len(z_s), len(y_r), len(y_s))
    z_r, z_s, y = z_r[:n], z_s[:n], y_r[:n]

    def score(name: str, z: np.ndarray) -> float:
        if len(np.unique(y)) < 2:
            return float("nan")
        return float(silhouette_score(z, y, metric="euclidean"))

    sil_r = score("random", z_r)
    sil_s = score("ssl", z_s)

    def umap_plot(z: np.ndarray, title: str, fname: str) -> None:
        if z.shape[0] < 15:
            return
        emb = UMAP(n_neighbors=min(15, z.shape[0] - 1), min_dist=0.1, metric="euclidean", random_state=42).fit_transform(z)
        plt.figure(figsize=(6, 5))
        plt.scatter(emb[:, 0], emb[:, 1], c=y, s=6, alpha=0.7, cmap="tab20")
        plt.colorbar()
        plt.title(title)
        plt.tight_layout()
        plt.savefig(out_dir / fname, dpi=150)
        plt.close()

    umap_plot(z_r, f"UMAP random backbone ({args.split})", "umap_random.png")
    umap_plot(z_s, f"UMAP SSL backbone ({args.split})", "umap_ssl.png")

    report = {
        "split": args.split,
        "n_points": int(n),
        "silhouette_random": sil_r,
        "silhouette_ssl": sil_s,
        "ssl_minus_random_silhouette": sil_s - sil_r if not (np.isnan(sil_s) or np.isnan(sil_r)) else None,
    }
    (out_dir / "representation_compare.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
