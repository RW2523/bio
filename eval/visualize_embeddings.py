#!/usr/bin/env python3
"""Optional 2D embedding visualization (UMAP preferred; t-SNE fallback)."""

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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--embeddings", type=str, required=True, help="Path to embeddings.npz with keys Z,y")
    ap.add_argument("--artifacts_dir", type=str, default="outputs/artifacts")
    ap.add_argument("--output", type=str, default="outputs/features/embedding_2d.png")
    ap.add_argument("--max_points", type=int, default=5000)
    ap.add_argument("--method", type=str, default="umap", choices=["umap", "tsne"])
    args = ap.parse_args()

    z = np.load(_resolve(args.embeddings))
    Z = z["Z"].astype(np.float32, copy=False)
    y = z["y"].astype(np.int64, copy=False)

    n = Z.shape[0]
    rng = np.random.default_rng(0)
    if n > args.max_points:
        idx = rng.choice(n, size=int(args.max_points), replace=False)
        Z = Z[idx]
        y = y[idx]

    if args.method == "umap":
        try:
            import umap  # type: ignore

            reducer = umap.UMAP(n_neighbors=30, min_dist=0.1, metric="cosine", random_state=0)
            xy = reducer.fit_transform(Z)
        except Exception as e:  # noqa: BLE001
            raise RuntimeError("UMAP failed; install `umap-learn` or use `--method tsne`.") from e
    else:
        from sklearn.manifold import TSNE

        xy = TSNE(n_components=2, init="pca", learning_rate="auto", perplexity=30, random_state=0).fit_transform(Z)

    label_map = read_json(_resolve(args.artifacts_dir) / "label_map.json")
    codes = list(label_map["index_to_code"])

    out_path = _resolve(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 8))
    for cls in sorted(set(y.tolist())):
        m = y == cls
        name = codes[cls] if 0 <= cls < len(codes) else str(cls)
        plt.scatter(xy[m, 0], xy[m, 1], s=6, label=name, alpha=0.7)
    plt.legend(markerscale=2, bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.0)
    plt.title(f"Embedding visualization ({args.method})")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


if __name__ == "__main__":
    main()
