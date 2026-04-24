"""Optional diagnostics for SSL / unfrozen backbone training."""

from __future__ import annotations

import torch.nn as nn


def backbone_grad_norm_l2(backbone: nn.Module) -> float:
    """Total L2 norm of gradients over all backbone parameters."""
    total = 0.0
    for p in backbone.parameters():
        if p.grad is not None:
            g = p.grad.detach().data
            total += float(g.norm(2).cpu() ** 2)
    return total ** 0.5


def read_spike_mean_from_snn(root: nn.Module) -> float | None:
    """Read ``_last_spike_mean`` from ``SpikingResNet1d`` inside ``EncodedBackbone`` or plain module."""
    from models.spiking_resnet1d import SpikingResNet1d

    if isinstance(root, SpikingResNet1d):
        v = getattr(root, "_last_spike_mean", None)
        return float(v) if v is not None else None
    if hasattr(root, "backbone"):
        return read_spike_mean_from_snn(root.backbone)  # type: ignore[arg-type]
    return None


def spike_mean_proxy_health_note(rate: float | None) -> str:
    """Short hint for logs; values are conv activation mean proxies, not literal spike rates."""
    if rate is None:
        return ""
    if rate < 0.05:
        return " [hint: very low proxy — risk dead pathway / high threshold / weak input]"
    if rate > 0.30:
        return " [hint: high proxy — risk saturation / low threshold]"
    return " [hint: typical exploratory band ~0.05–0.30 for this proxy]"
