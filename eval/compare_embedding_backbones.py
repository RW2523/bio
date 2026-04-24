#!/usr/bin/env python3
"""
Side-by-side 2D embedding comparison for two frozen-backbone feature sets.

Typical inputs (same --split so rows align and y matches):
  1) Random backbone: Case 1 probe checkpoint (backbone never trained):
       python eval/extract_features.py --backbone_checkpoint outputs/checkpoints/case1/best.pt \\
         --split train --output outputs/features/embeddings_train_random.npz
  2) SSL backbone:
       python eval/extract_features.py --backbone_checkpoint outputs/checkpoints/ssl/backbone_best.pt \\
         --split train --output outputs/features/embeddings_train_ssl.npz

Then:
  python eval/compare_embedding_backbones.py \\
    --embeddings_random outputs/features/embeddings_train_random.npz \\
    --embeddings_pretrained outputs/features/embeddings_train_ssl.npz \\
    --output outputs/features/compare_train_umap.png \\
    --method umap
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.io import read_json
from utils.paths import project_root


def _resolve(p: str) -> Path:
    pp = Path(p)
    return pp if pp.is_absolute() else (project_root() / pp).resolve()


def _umap_2d(Z: np.ndarray, *, random_state: int = 0) -> np.ndarray:
    import umap  # type: ignore

    reducer = umap.UMAP(n_neighbors=30, min_dist=0.1, metric="cosine", random_state=random_state)
    return reducer.fit_transform(Z.astype(np.float32, copy=False))


def _tsne_2d(Z: np.ndarray, *, random_state: int = 0) -> np.ndarray:
    from sklearn.manifold import TSNE

    return TSNE(
        n_components=2,
        init="pca",
        learning_rate="auto",
        perplexity=30,
        random_state=random_state,
    ).fit_transform(Z.astype(np.float32, copy=False))


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare two embedding NPZs (same windows / same y).")
    ap.add_argument("--embeddings_random", type=str, required=True, help="NPZ from extract_features (random init backbone)")
    ap.add_argument("--embeddings_pretrained", type=str, required=True, help="NPZ from extract_features (SSL backbone)")
    ap.add_argument("--artifacts_dir", type=str, default="outputs/artifacts", help="For activity code legend")
    ap.add_argument("--output", type=str, default="outputs/features/compare_embeddings.png")
    ap.add_argument("--method", type=str, default="umap", choices=["umap", "tsne"])
    ap.add_argument("--max_points", type=int, default=8000, help="Subsample per panel if larger (same idx for both)")
    ap.add_argument("--sample_seed", type=int, default=0)
    args = ap.parse_args()

    pa = _resolve(args.embeddings_random)
    pb = _resolve(args.embeddings_pretrained)
    za = np.load(pa)
    zb = np.load(pb)
    Zr = za["Z"].astype(np.float32, copy=False)
    Zp = zb["Z"].astype(np.float32, copy=False)
    yr = za["y"].astype(np.int64, copy=False)
    yp = zb["y"].astype(np.int64, copy=False)

    if Zr.shape != Zp.shape or yr.shape != yp.shape:
        raise ValueError(f"Shape mismatch: random {Zr.shape} vs pretrained {Zp.shape}")
    if not np.array_equal(yr, yp):
        raise ValueError("Label arrays differ; extract both with the same --split and dataset state.")

    n = Zr.shape[0]
    rng = np.random.default_rng(int(args.sample_seed))
    if n > int(args.max_points):
        idx = rng.choice(n, size=int(args.max_points), replace=False)
        Zr, Zp, y = Zr[idx], Zp[idx], yr[idx]
    else:
        y = yr

    label_map = read_json(_resolve(args.artifacts_dir) / "label_map.json")
    codes = list(label_map["index_to_code"])

    # High-dimensional class separation (same labels; comparable across Zr vs Zp)
    try:
        from sklearn.metrics import silhouette_score

        # Subsample for speed if huge
        max_sil = min(5000, Zr.shape[0])
        if Zr.shape[0] > max_sil:
            si = rng.choice(Zr.shape[0], size=max_sil, replace=False)
            sil_r = float(silhouette_score(Zr[si], y[si], metric="euclidean"))
            sil_p = float(silhouette_score(Zp[si], y[si], metric="euclidean"))
        else:
            sil_r = float(silhouette_score(Zr, y, metric="euclidean"))
            sil_p = float(silhouette_score(Zp, y, metric="euclidean"))
    except Exception as e:  # noqa: BLE001
        sil_r, sil_p = float("nan"), float("nan")
        sil_note = f" (silhouette skipped: {e})"
    else:
        sil_note = ""

    reducer = _umap_2d if args.method == "umap" else _tsne_2d
    xy_r = reducer(Zr)
    xy_p = reducer(Zp)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=False, sharey=False)
    titles = (
        f"Random init backbone ({args.method})\nsilhouette≈{sil_r:.3f}{sil_note}",
        f"SSL pretrained backbone ({args.method})\nsilhouette≈{sil_p:.3f}",
    )
    for ax, xy, title in zip(axes, (xy_r, xy_p), titles):
        for cls in sorted(set(y.tolist())):
            m = y == cls
            name = codes[cls] if 0 <= cls < len(codes) else str(cls)
            ax.scatter(xy[m, 0], xy[m, 1], s=5, label=name, alpha=0.65)
        ax.set_title(title)
        ax.set_xlabel("dim-0")
        ax.set_ylabel("dim-1")
    axes[1].legend(markerscale=2, bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    fig.suptitle("Frozen backbone embeddings (separate 2D fit per panel; same window subset)", fontsize=11)
    fig.tight_layout()
    out = _resolve(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")
    print(f"silhouette_euclidean_random={sil_r:.4f} pretrained={sil_p:.4f}")


if __name__ == "__main__":
    main()
