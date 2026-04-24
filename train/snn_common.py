"""Shared builders for configurable SNN frontends/backbones."""

from __future__ import annotations

from typing import Any, Mapping

import torch.nn as nn

from datasets.window_features import infer_in_channels_from_feature_stack, normalize_feature_stack
from models.encoded_backbone import EncodedBackbone
from models.encoders import IdentityEncoder, ZCSFEncoder
from models.spiking_resnet1d import SpikingResNet1d
from train.temporal_pool_cfg import canonical_temporal_pooling, temporal_pooling_from_yaml


def snn_model_cfg_from_yaml(cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Subset of YAML needed to rebuild the SNN stack from checkpoints."""
    fs = normalize_feature_stack(cfg.get("feature_stack"))
    base_sensor_c = int(cfg.get("sensor_channels", 3))
    infer_ch = infer_in_channels_from_feature_stack(fs, base_channels=base_sensor_c)
    if cfg.get("in_channels") is not None and int(cfg["in_channels"]) != infer_ch:
        raise ValueError(
            f"in_channels={cfg.get('in_channels')} inconsistent with feature_stack={fs} "
            f"(expected {infer_ch} for sensor_channels={base_sensor_c})."
        )
    dbg = cfg.get("debug")
    dbg_dict = dict(dbg) if isinstance(dbg, dict) else {}
    pool_mode, pool_bins = temporal_pooling_from_yaml(cfg)
    return {
        "backbone_type": "spiking_resnet1d",
        "input_encoder_type": str(cfg.get("input_encoder_type", "identity")),
        "input_encoder_cfg": {
            "step": float(cfg.get("zcsf_step", 0.25)),
            "emit_start_event": bool(cfg.get("zcsf_emit_start_event", False)),
        },
        "feature_stack": fs,
        "sensor_channels": base_sensor_c,
        "in_channels": infer_ch,
        "base_channels": int(cfg.get("base_channels", 32)),
        "layers": [int(x) for x in cfg.get("layers", [1, 1, 2])],
        "stem_kernel": int(cfg.get("stem_kernel", 7)),
        "lif_beta": float(cfg.get("lif_beta", 0.9)),
        "lif_threshold": float(cfg.get("lif_threshold", 1.0)),
        "surrogate_alpha": float(cfg.get("surrogate_alpha", 2.0)),
        "lif_reset": str(cfg.get("lif_reset", "subtract")),
        "pooling": pool_mode,
        "pool_bins": int(pool_bins),
        "debug": dbg_dict,
    }


def build_input_encoder(model_cfg: Mapping[str, Any]) -> nn.Module:
    """Create the configured sensor-to-spike encoder (after optional feature stacking)."""
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


def merge_snn_model_cfg_from_checkpoint(
    ckpt_model_cfg: Mapping[str, Any] | None,
    yaml_cfg: Mapping[str, Any],
) -> dict[str, Any]:
    """
    Merge YAML defaults with checkpoint ``model_cfg`` (checkpoint wins on overlap).

    Older SSL checkpoints omit ``feature_stack``; those runs used raw xyz only, so we default
    ``feature_stack`` to ``['raw']`` when absent rather than inheriting probe-only YAML.
    """
    defaults = snn_model_cfg_from_yaml(yaml_cfg)
    if not ckpt_model_cfg:
        return defaults
    ck = dict(ckpt_model_cfg)
    merged = {**defaults, **ck}
    if "feature_stack" not in ck:
        merged["feature_stack"] = normalize_feature_stack(None)
    merged["feature_stack"] = normalize_feature_stack(merged.get("feature_stack"))
    base_c = int(merged.get("sensor_channels", 3))
    merged["in_channels"] = infer_in_channels_from_feature_stack(merged["feature_stack"], base_channels=base_c)
    p, b = canonical_temporal_pooling(merged.get("pooling", "mean"), merged.get("pool_bins", 1))
    merged["pooling"], merged["pool_bins"] = p, int(b)
    return merged


def build_snn_backbone(model_cfg: Mapping[str, Any]) -> nn.Module:
    """Build the full SNN stack, optionally with a front-end encoder."""
    encoder = build_input_encoder(model_cfg)
    dbg = model_cfg.get("debug") or {}
    spike_dbg = bool(dbg.get("log_spike_rate", False)) if isinstance(dbg, dict) else False
    backbone = SpikingResNet1d(
        in_channels=int(encoder.out_channels),
        base_channels=int(model_cfg["base_channels"]),
        layers=tuple(int(x) for x in model_cfg["layers"]),
        stem_kernel=int(model_cfg["stem_kernel"]),
        beta=float(model_cfg["lif_beta"]),
        threshold=float(model_cfg["lif_threshold"]),
        surrogate_alpha=float(model_cfg["surrogate_alpha"]),
        reset=str(model_cfg["lif_reset"]),
        pooling=str(model_cfg.get("pooling", "mean")).strip().lower(),
        pool_bins=int(model_cfg.get("pool_bins", 1)),
        debug_log_spike_rate=spike_dbg,
    )
    if encoder.__class__.__name__ == "IdentityEncoder":
        return backbone
    return EncodedBackbone(encoder, backbone)
