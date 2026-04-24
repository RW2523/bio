"""Shared helpers for CNN baseline training/evaluation scripts."""

from __future__ import annotations

from typing import Any, Mapping

from datasets.window_features import infer_in_channels_from_feature_stack, normalize_feature_stack
from models.cnn_resnet1d import CNNResNet1d
from train.temporal_pool_cfg import canonical_temporal_pooling, temporal_pooling_from_yaml


def cnn_model_cfg_from_yaml(cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Subset of YAML needed to rebuild the CNN backbone exactly from checkpoints."""
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
        "backbone_type": "cnn_resnet1d",
        "feature_stack": fs,
        "sensor_channels": base_sensor_c,
        "in_channels": infer_ch,
        "base_channels": int(cfg.get("base_channels", 32)),
        "layers": [int(x) for x in cfg.get("layers", [1, 1, 2])],
        "stem_kernel": int(cfg.get("stem_kernel", 7)),
        "pooling": pool_mode,
        "pool_bins": int(pool_bins),
        "debug": dbg_dict,
    }


def merge_cnn_model_cfg_from_checkpoint(
    ckpt_model_cfg: Mapping[str, Any] | None,
    yaml_cfg: Mapping[str, Any],
) -> dict[str, Any]:
    defaults = cnn_model_cfg_from_yaml(yaml_cfg)
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


def build_cnn_backbone(model_cfg: Mapping[str, Any]) -> CNNResNet1d:
    """Construct the continuous CNN ResNet backbone from a saved/config dict."""
    return CNNResNet1d(
        in_channels=int(model_cfg["in_channels"]),
        base_channels=int(model_cfg["base_channels"]),
        layers=tuple(int(x) for x in model_cfg["layers"]),
        stem_kernel=int(model_cfg["stem_kernel"]),
        pooling=str(model_cfg.get("pooling", "mean")).strip().lower(),
        pool_bins=int(model_cfg.get("pool_bins", 1)),
    )
