#!/usr/bin/env python3
"""
Lightweight end-to-end smoke checks (no full training).

Run from repository root:
  python verify/smoke_e2e.py
"""

from __future__ import annotations

import compileall
import sys
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _check_compileall() -> None:
    for sub in ("configs", "data_tools", "datasets", "eval", "models", "train", "transforms", "utils", "verify"):
        p = ROOT / sub
        if p.is_dir():
            if not compileall.compile_dir(str(p), quiet=1):
                raise RuntimeError(f"compileall failed under {p}")


def _check_imports() -> None:
    import data_tools.build_manifest  # noqa: F401
    import data_tools.inspect_dataset  # noqa: F401
    import data_tools.preprocess_wisdm  # noqa: F401
    import datasets.wisdm_ssl_dataset  # noqa: F401
    import datasets.wisdm_supervised_dataset  # noqa: F401
    import eval.evaluate  # noqa: F401
    import models.spiking_resnet1d  # noqa: F401
    import train.pretrain_augpred  # noqa: F401
    import train.train_linear_probe_case1  # noqa: F401
    import train.train_linear_probe_case2  # noqa: F401
    import transforms.augpred  # noqa: F401


def _check_surrogate_gradient() -> None:
    from models.surrogate import spike_fn

    x = torch.randn(4, requires_grad=True)
    s = spike_fn(x - 0.1, alpha=2.0)
    loss = s.sum()
    loss.backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert float(x.grad.abs().sum()) > 0.0


def _check_spiking_forward() -> None:
    from models.spiking_resnet1d import SpikingResNet1d

    m = SpikingResNet1d(in_channels=3, base_channels=16, layers=(1, 1), stem_kernel=5)
    x = torch.randn(2, 3, 64)
    z = m(x)
    assert z.shape == (2, m.out_dim)


def _check_ssl_heads() -> None:
    from models.spiking_resnet1d import SpikingResNet1d
    from models.ssl_heads import AugPredSSLHeads
    from transforms.augpred import sample_arrow_of_time, sample_permutation, sample_time_warp

    bb = SpikingResNet1d(in_channels=3, base_channels=16, layers=(1,), stem_kernel=5)
    heads = AugPredSSLHeads(bb.out_dim)
    x = torch.randn(3, 32, 3)
    crit = nn.BCEWithLogitsLoss()
    xa, ya = sample_arrow_of_time(x)
    z = bb(xa.transpose(1, 2))
    la = crit(heads.head_aot(z).squeeze(-1), ya)
    xp, yp = sample_permutation(x, 4)
    zp = bb(xp.transpose(1, 2))
    lp = crit(heads.head_perm(zp).squeeze(-1), yp)
    xw, yw = sample_time_warp(x, 0.15)
    zw = bb(xw.transpose(1, 2))
    lw = crit(heads.head_tw(zw).squeeze(-1), yw)
    loss = la + lp + lw
    loss.backward()
    assert torch.isfinite(loss)


def _check_probe_optimizer() -> None:
    from models.linear_probe import LinearProbeHead
    from models.spiking_resnet1d import SpikingResNet1d
    from train.common import assert_optimizer_excludes_module, freeze_module

    bb = SpikingResNet1d(in_channels=3, base_channels=8, layers=(1,), stem_kernel=3)
    freeze_module(bb)
    head = LinearProbeHead(bb.out_dim, num_classes=5)
    opt = torch.optim.SGD(head.parameters(), lr=0.1)
    assert_optimizer_excludes_module(opt, bb)


def main() -> None:
    print("compileall...")
    _check_compileall()
    print("imports...")
    _check_imports()
    print("surrogate grad...")
    _check_surrogate_gradient()
    print("spiking resnet forward...")
    _check_spiking_forward()
    print("ssl heads + augpred backward...")
    _check_ssl_heads()
    print("probe optimizer guard...")
    _check_probe_optimizer()
    print("OK: verify/smoke_e2e.py passed.")


if __name__ == "__main__":
    main()
