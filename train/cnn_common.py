"""Shared helpers for CNN baseline training/evaluation scripts."""

from __future__ import annotations

from typing import Any, Mapping

from models.cnn_resnet1d import CNNResNet1d


def cnn_model_cfg_from_yaml(cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Subset of YAML needed to rebuild the CNN backbone exactly from checkpoints."""
    return {
        "in_channels": int(cfg.get("in_channels", 3)),
        "base_channels": int(cfg.get("base_channels", 32)),
        "layers": [int(x) for x in cfg.get("layers", [1, 1, 2])],
        "stem_kernel": int(cfg.get("stem_kernel", 7)),
    }


def build_cnn_backbone(model_cfg: Mapping[str, Any]) -> CNNResNet1d:
    """Construct the continuous CNN ResNet backbone from a saved/config dict."""
    return CNNResNet1d(
        in_channels=int(model_cfg["in_channels"]),
        base_channels=int(model_cfg["base_channels"]),
        layers=tuple(int(x) for x in model_cfg["layers"]),
        stem_kernel=int(model_cfg["stem_kernel"]),
    )
