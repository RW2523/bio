"""Save learning-curve figures from per-epoch metrics (no training logic)."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def save_supervised_curves_png(curves: Mapping[str, list[float]], out_path: Path, title: str = "") -> None:
    """
    Plot train/val loss and train/val accuracy when present.

    Expected keys (all optional except at least one series):
    ``train_loss``, ``val_loss``, ``train_acc``, ``val_acc``.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    n = max(
        len(curves.get("train_loss", [])),
        len(curves.get("val_loss", [])),
        len(curves.get("train_acc", [])),
        len(curves.get("val_acc", [])),
    )
    if n == 0:
        plt.close(fig)
        return
    epochs = list(range(1, n + 1))

    ax0 = axes[0]
    if curves.get("train_loss"):
        ax0.plot(epochs[: len(curves["train_loss"])], curves["train_loss"], label="train_loss")
    if curves.get("val_loss"):
        ax0.plot(epochs[: len(curves["val_loss"])], curves["val_loss"], label="val_loss")
    ax0.set_xlabel("epoch")
    ax0.set_ylabel("loss")
    ax0.legend()
    ax0.grid(True, alpha=0.3)

    ax1 = axes[1]
    if curves.get("train_acc"):
        ax1.plot(epochs[: len(curves["train_acc"])], curves["train_acc"], label="train_acc")
    if curves.get("val_acc"):
        ax1.plot(epochs[: len(curves["val_acc"])], curves["val_acc"], label="val_acc")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("accuracy")
    ax1.set_ylim(0.0, 1.0)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    if title:
        fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_ssl_loss_curves_png(curves: Mapping[str, list[float]], out_path: Path, title: str = "") -> None:
    """Plot SSL pretraining ``train_loss`` and ``val_loss`` (no classification accuracy)."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    tl = curves.get("train_loss") or []
    vl = curves.get("val_loss") or []
    n = max(len(tl), len(vl))
    if n == 0:
        return
    epochs = list(range(1, n + 1))
    fig, ax = plt.subplots(figsize=(6, 4))
    if tl:
        ax.plot(epochs[: len(tl)], tl, label="train_loss")
    if vl:
        ax.plot(epochs[: len(vl)], vl, label="val_loss")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    if title:
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
