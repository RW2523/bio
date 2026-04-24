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

import numpy as np
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


def _check_feature_stack_channel_counts() -> None:
    from datasets.window_features import infer_input_channels, stack_window_features

    x = np.random.randn(48, 3).astype(np.float32)
    assert stack_window_features(x, ["raw"]).shape == (48, 3)
    assert stack_window_features(x, ["raw", "delta"]).shape == (48, 6)
    assert stack_window_features(x, ["raw", "delta", "window_std"]).shape == (48, 9)
    assert infer_input_channels(["raw"], 3) == 3
    assert infer_input_channels(["raw", "delta"], 3) == 6
    assert infer_input_channels(["raw", "delta", "window_std"], 3) == 9


def _check_pooling_out_dim_and_linear_probe() -> None:
    from models.cnn_resnet1d import CNNResNet1d
    from models.linear_probe import LinearProbeHead
    from models.spiking_resnet1d import SpikingResNet1d

    for pooling, pool_bins in (("mean", 1), ("adaptive", 4), ("attention", 1)):
        for in_ch in (3, 6, 9):
            cnn = CNNResNet1d(
                in_channels=in_ch,
                base_channels=16,
                layers=(1, 1),
                stem_kernel=5,
                pooling=pooling,
                pool_bins=pool_bins,
            )
            snn = SpikingResNet1d(
                in_channels=in_ch,
                base_channels=16,
                layers=(1, 1),
                stem_kernel=5,
                pooling=pooling,
                pool_bins=pool_bins,
            )
            x = torch.randn(2, in_ch, 64)
            zc, zs = cnn(x), snn(x)
            assert zc.shape == (2, cnn.out_dim) == (2, snn.out_dim)
            assert zs.shape == zc.shape
            head_c = LinearProbeHead(cnn.out_dim, num_classes=7)
            head_s = LinearProbeHead(snn.out_dim, num_classes=7)
            assert head_c(zc).shape == (2, 7)
            assert head_s(zs).shape == (2, 7)


def _check_adaptive4_yaml_canonical() -> None:
    from train.snn_common import snn_model_cfg_from_yaml
    from utils.yaml_config import load_merged_config

    m = dict(load_merged_config("model"))
    m["pooling"] = "adaptive4"
    m["pool_bins"] = 99
    mc = snn_model_cfg_from_yaml(m)
    assert mc["pooling"] == "adaptive" and mc["pool_bins"] == 4


def _check_mlp_head_and_motion_ce() -> None:
    from train.common import build_probe_criterion, linear_probe_head_from_cfg, motion_weighted_cross_entropy

    h = linear_probe_head_from_cfg(
        32, 5, {"classifier_head": "mlp", "classifier_dropout": 0.3, "probe_embedding_batchnorm": False}
    )
    assert h(torch.randn(3, 32)).shape == (3, 5)
    crit = build_probe_criterion(5, torch.device("cpu"), {}, class_counts=None)
    loss = motion_weighted_cross_entropy(
        torch.randn(4, 5),
        torch.tensor([0, 1, 2, 3]),
        torch.tensor([1.0, 2.0, 3.0, 4.0]),
        alpha=1.0,
        crit=crit,
    )
    assert loss.ndim == 0 and torch.isfinite(loss)


def _check_supervised_dataset_batch_shapes_and_backbones() -> None:
    """One cached subject + one batch: channel counts and CNN/SNN embedding vs linear head."""
    import tempfile

    from datasets.wisdm_supervised_dataset import WISDMSupervisedDataset
    from train.cnn_common import build_cnn_backbone, cnn_model_cfg_from_yaml
    from train.common import linear_probe_head_from_cfg, to_bct
    from train.snn_common import build_snn_backbone, snn_model_cfg_from_yaml
    from utils.yaml_config import load_merged_config

    td = tempfile.mkdtemp()
    cache = Path(td)
    t, c, n = 32, 3, 2
    xraw = np.random.randn(n, t, c).astype(np.float32)
    np.savez_compressed(
        cache / "subject_7.npz",
        X=xraw,
        y=np.zeros(n, dtype=np.int64),
        motion=np.ones(n, dtype=np.float64),
    )
    mean = np.zeros(c, dtype=np.float32)
    std = np.ones(c, dtype=np.float32)

    stacks = (
        (["raw"], 3),
        (["raw", "delta"], 6),
        (["raw", "delta", "window_std"], 9),
    )
    base_yaml = load_merged_config("model")
    for fs, expect_c in stacks:
        cfg = dict(base_yaml)
        cfg["feature_stack"] = list(fs)
        cfg["in_channels"] = int(expect_c)
        cfg["pooling"] = "adaptive"
        cfg["pool_bins"] = 2
        ds = WISDMSupervisedDataset(cache, [7], mean=mean, std=std, feature_stack=fs)
        loader = torch.utils.data.DataLoader(ds, batch_size=n, shuffle=False)
        x_btc, y, _w = next(iter(loader))
        assert tuple(x_btc.shape) == (n, t, expect_c), (x_btc.shape, expect_c)
        print(f"  supervised batch x shape {tuple(x_btc.shape)} (expect C={expect_c})")

        mc_s = snn_model_cfg_from_yaml(cfg)
        mc_c = cnn_model_cfg_from_yaml(cfg)
        bb_s = build_snn_backbone(mc_s)
        bb_c = build_cnn_backbone(mc_c)
        x_bct = to_bct(x_btc)
        zs = bb_s(x_bct)
        zc = bb_c(x_bct)
        assert zs.shape == (n, bb_s.out_dim) == (n, bb_c.out_dim)
        head_s = linear_probe_head_from_cfg(bb_s.out_dim, 5, {"probe_embedding_batchnorm": False})
        head_c = linear_probe_head_from_cfg(bb_c.out_dim, 5, {"probe_embedding_batchnorm": False})
        assert head_s(zs).shape == (n, 5)
        assert head_c(zc).shape == (n, 5)


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
    print("feature_stack channel shapes...")
    _check_feature_stack_channel_counts()
    print("pooling out_dim + linear probe...")
    _check_pooling_out_dim_and_linear_probe()
    print("adaptive4 yaml canonicalization...")
    _check_adaptive4_yaml_canonical()
    print("mlp head + motion-weighted CE...")
    _check_mlp_head_and_motion_ce()
    print("supervised dataset one batch + CNN/SNN...")
    _check_supervised_dataset_batch_shapes_and_backbones()
    print("OK: verify/smoke_e2e.py passed.")


if __name__ == "__main__":
    main()
