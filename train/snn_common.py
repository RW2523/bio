"""Shared builders for configurable SNN frontends/backbones."""

from __future__ import annotations

from typing import Any, Mapping

import torch.nn as nn

from models.encoded_backbone import EncodedBackbone
from models.encoders import IdentityEncoder, ZCSFEncoder
from models.spiking_resnet1d import SpikingResNet1d


def snn_model_cfg_from_yaml(cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Subset of YAML needed to rebuild the SNN stack from checkpoints."""
    return {
        "backbone_type": "spiking_resnet1d",
        "input_encoder_type": str(cfg.get("input_encoder_type", "identity")),
        "input_encoder_cfg": {
            "step": float(cfg.get("zcsf_step", 0.25)),
            "emit_start_event": bool(cfg.get("zcsf_emit_start_event", False)),
        },
        "in_channels": int(cfg.get("in_channels", 3)),
        "base_channels": int(cfg.get("base_channels", 32)),
        "layers": [int(x) for x in cfg.get("layers", [1, 1, 2])],
        "stem_kernel": int(cfg.get("stem_kernel", 7)),
        "lif_beta": float(cfg.get("lif_beta", 0.9)),
        "lif_threshold": float(cfg.get("lif_threshold", 1.0)),
        "surrogate_alpha": float(cfg.get("surrogate_alpha", 2.0)),
        "lif_reset": str(cfg.get("lif_reset", "subtract")),
    }


def build_input_encoder(model_cfg: Mapping[str, Any]) -> nn.Module:
    """Create the configured sensor-to-spike encoder."""
    in_channels = int(model_cfg["in_channels"])
    enc_type = str(model_cfg.get("input_encoder_type", "identity")).strip().lower()
    enc_cfg = dict(model_cfg.get("input_encoder_cfg") or {})
    if enc_type in {"", "identity", "none"}:
        return IdentityEncoder(channels=in_channels)
    if enc_type == "zcsf":
        return ZCSFEncoder(
            channels=in_channels,
            step=float(enc_cfg.get("step", 0.25)),
            emit_start_event=bool(enc_cfg.get("emit_start_event", False)),
        )
    raise ValueError(f"Unknown input_encoder_type {enc_type!r}")


def build_snn_backbone(model_cfg: Mapping[str, Any]) -> nn.Module:
    """Build the full SNN stack, optionally with a front-end encoder."""
    encoder = build_input_encoder(model_cfg)
    backbone = SpikingResNet1d(
        in_channels=int(encoder.out_channels),
        base_channels=int(model_cfg["base_channels"]),
        layers=tuple(int(x) for x in model_cfg["layers"]),
        stem_kernel=int(model_cfg["stem_kernel"]),
        beta=float(model_cfg["lif_beta"]),
        threshold=float(model_cfg["lif_threshold"]),
        surrogate_alpha=float(model_cfg["surrogate_alpha"]),
        reset=str(model_cfg["lif_reset"]),
    )
    if encoder.__class__.__name__ == "IdentityEncoder":
        return backbone
    return EncodedBackbone(encoder, backbone)
