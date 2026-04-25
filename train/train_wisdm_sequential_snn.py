#!/usr/bin/env python3
"""
Sequential reservoir SNN + TR encoding + STDP (unsupervised) + supervised head for WISDM.

Uses the same window cache, subject splits, and train-only normalization as the spiking ResNet path.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Mapping

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.wisdm_supervised_dataset import WISDMSupervisedDataset
from eval.confusion_matrix import save_confusion_matrix_figure
from eval.metrics import compute_metrics
from models.desnn_classifier import build_output_head
from models.encoders.tr_spike_encoder import TRSpikeEncoder
from models.sequential_snn_reservoir import ReservoirGeometry, SequentialSNNReservoir
from models.stdp import STDPConfig
from train.common import (
    balanced_class_weights,
    build_probe_lr_scheduler,
    load_label_map,
    load_norm_stats,
    load_splits,
    make_loader,
    resolve_compute_device,
    resolve_path,
    subject_split_ids,
)
from utils.assertions import assert_disjoint_subject_sets, assert_label_range
from utils.checkpoint import save_checkpoint
from utils.io import read_json, write_json
from utils.logger import setup_logger
from utils.seed import set_seed
from utils.yaml_config import load_merged_config


def _infer_cache_channels(cache_dir: Path, subject_ids: list[int]) -> int:
    for sid in subject_ids:
        p = cache_dir / f"subject_{sid}.npz"
        if p.exists():
            z = np.load(p)
            return int(z["X"].shape[2])
    raise FileNotFoundError(f"No cached subject npz under {cache_dir}")


def _effective_channels(
    cache_channels: int,
    use_gyro: bool,
) -> int:
    if cache_channels == 3:
        return 3
    if cache_channels == 6:
        return 6 if use_gyro else 3
    return cache_channels


def _slice_batch_x(x: torch.Tensor, *, cache_channels: int, use_gyro: bool) -> torch.Tensor:
    if cache_channels == 6 and not use_gyro:
        return x[..., :3].contiguous()
    return x


def _norm_for_channels(norm, cache_channels: int, use_gyro: bool):
    mean, std = norm.mean, norm.std
    if cache_channels == 6 and not use_gyro:
        return mean[:3], std[:3]
    return mean, std


def _assert_spike_shape(spikes: torch.Tensor, b: int, t_exp: int, f: int) -> None:
    if spikes.ndim != 3 or spikes.shape[0] != b or spikes.shape[1] != t_exp or spikes.shape[2] != f:
        raise AssertionError(f"Unexpected spike shape {tuple(spikes.shape)} expected [*,{t_exp},{f}]")


def _save_training_curves(curves: dict[str, list[float]], out_path: Path, title: str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(curves["train_loss"], label="train")
    axes[0].plot(curves["val_loss"], label="val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("epoch")
    axes[0].legend()
    axes[1].plot(curves["train_acc"], label="train")
    axes[1].plot(curves["val_acc"], label="val")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("epoch")
    axes[1].legend()
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _epoch_classifier(
    encoder: TRSpikeEncoder,
    reservoir: SequentialSNNReservoir,
    head: nn.Module,
    loader: DataLoader,
    device: torch.device,
    crit: nn.Module,
    opt: torch.optim.Optimizer,
    num_classes: int,
    *,
    train: bool,
    cache_channels: int,
    use_gyro: bool,
    max_batches: int | None = None,
    feature_mixup_alpha: float = 0.0,
) -> tuple[float, float]:
    if train:
        head.train()
    else:
        head.eval()
    encoder.eval()
    reservoir.eval()
    tot = 0.0
    m = 0
    correct = 0
    count = 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for i, (x, y, _w) in enumerate(loader):
            if max_batches is not None and i >= max_batches:
                break
            x = _slice_batch_x(x.to(device), cache_channels=cache_channels, use_gyro=use_gyro)
            y = y.to(device)
            assert_label_range(y, num_classes)
            with torch.no_grad():
                spikes = encoder(x)
            with torch.no_grad():
                feats, _ = reservoir(spikes, stdp=False)
            if train:
                opt.zero_grad(set_to_none=True)
            mix_a = float(feature_mixup_alpha)
            if train and mix_a > 0.0 and feats.size(0) >= 2:
                dist = torch.distributions.Beta(
                    torch.tensor(mix_a, device=device, dtype=torch.float32),
                    torch.tensor(mix_a, device=device, dtype=torch.float32),
                )
                lam = float(dist.sample().item())
                lam = min(max(lam, 0.05), 0.95)
                perm = torch.randperm(feats.size(0), device=device)
                feats_m = lam * feats + (1.0 - lam) * feats[perm]
                logits = head(feats_m)
                y_perm = y[perm]
                loss = lam * crit(logits, y) + (1.0 - lam) * crit(logits, y_perm)
            else:
                logits = head(feats)
                loss = crit(logits, y)
            if train:
                loss.backward()
                opt.step()
            tot += float(loss.detach().cpu())
            m += 1
            pred = torch.argmax(logits, dim=-1)
            correct += int((pred == y).sum().item())
            count += int(y.numel())
    return tot / max(m, 1), correct / max(count, 1)


def _payload_from_parts(
    *,
    encoder_cfg: dict[str, Any],
    reservoir_cfg: dict[str, Any],
    stdp_cfg: dict[str, Any],
    head_mode: str,
    head_cfg: dict[str, Any],
    encoder: TRSpikeEncoder,
    reservoir: SequentialSNNReservoir,
    head: nn.Module,
    num_classes: int,
    meta: dict[str, Any],
) -> dict[str, Any]:
    return {
        "kind": "sequential_snn_wisdm",
        "encoder_cfg": encoder_cfg,
        "reservoir_cfg": reservoir_cfg,
        "stdp_cfg": stdp_cfg,
        "classifier_mode": head_mode,
        "head_cfg": head_cfg,
        "encoder_state": encoder.state_dict(),
        "reservoir": reservoir.state_dict(),
        "head": head.state_dict(),
        "num_classes": int(num_classes),
        "meta": meta,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="sequential_snn_wisdm")
    args = ap.parse_args()

    cfg = load_merged_config(args.config)
    set_seed(int(cfg.get("seed", 42)))

    out_root = resolve_path(cfg["output_dir"])
    art_dir = out_root / str(cfg["artifacts_subdir"])
    cache_dir = out_root / str(cfg["cache_subdir"])
    ckpt_dir = out_root / str(cfg.get("sequential_checkpoint_subdir", "checkpoints/sequential_snn"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = ckpt_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(log_file=ckpt_dir / "train_sequential_snn.log")
    device = resolve_compute_device(str(cfg.get("compute_device", "cuda")))
    logger.info("Compute device: %s", device)

    splits = load_splits(art_dir)
    train_ids, val_ids, test_ids = subject_split_ids(splits)
    assert_disjoint_subject_sets(train_ids, val_ids, test_ids)

    dbg_max_tr = cfg.get("debug_max_train_subjects")
    dbg_max_va = cfg.get("debug_max_val_subjects")
    if dbg_max_tr is not None:
        train_ids = train_ids[: int(dbg_max_tr)]
    if dbg_max_va is not None:
        val_ids = val_ids[: int(dbg_max_va)]

    label_map = load_label_map(art_dir)
    num_classes = int(label_map["num_classes"])
    base_norm = load_norm_stats(art_dir)

    cache_channels = _infer_cache_channels(cache_dir, train_ids or val_ids or test_ids)
    use_gyro = bool(cfg.get("use_gyro", False))
    eff_c = _effective_channels(cache_channels, use_gyro)
    mean, std = _norm_for_channels(base_norm, cache_channels, use_gyro)
    if int(mean.shape[0]) != eff_c:
        raise ValueError(
            f"norm_stats channel count {mean.shape[0]} incompatible with effective_channels={eff_c} "
            f"(cache_channels={cache_channels}, use_gyro={use_gyro})."
        )

    train_ds = WISDMSupervisedDataset(cache_dir, train_ids, mean=mean, std=std)
    val_ds = WISDMSupervisedDataset(cache_dir, val_ids, mean=mean, std=std)

    train_loader = make_loader(
        train_ds,
        batch_size=int(cfg.get("stdp_batch_size", 1)),
        shuffle=True,
        num_workers=int(cfg.get("num_workers", 2)),
        device=device,
    )
    clf_bs = int(cfg.get("classifier_batch_size", 64))
    train_loader_clf = make_loader(
        train_ds,
        batch_size=clf_bs,
        shuffle=True,
        num_workers=int(cfg.get("num_workers", 2)),
        device=device,
    )
    val_loader = make_loader(
        val_ds,
        batch_size=clf_bs,
        shuffle=False,
        num_workers=int(cfg.get("num_workers", 2)),
        device=device,
    )

    window_samples = int(read_json(art_dir / "preprocess_run.json")["window_samples"])

    enc_cfg = cfg.get("tr_encoder", {})
    encoder = TRSpikeEncoder(
        num_channels=eff_c,
        threshold=float(enc_cfg.get("threshold", 0.25)),
        threshold_scale=str(enc_cfg.get("threshold_scale", "absolute")),
        encoding=str(enc_cfg.get("encoding", "two_channel")),
        delta_std=None,
    ).to(device)

    res_cfg = cfg.get("reservoir", {})
    geom = ReservoirGeometry(
        nx=int(res_cfg.get("nx", 4)),
        ny=int(res_cfg.get("ny", 4)),
        nz=int(res_cfg.get("nz", 4)),
    )
    stdp_yaml = cfg.get("stdp", {})
    stdp_cfg = STDPConfig(
        tau_plus=float(stdp_yaml.get("tau_plus", 20.0)),
        tau_minus=float(stdp_yaml.get("tau_minus", 20.0)),
        a_plus=float(stdp_yaml.get("a_plus", 0.004)),
        a_minus=float(stdp_yaml.get("a_minus", 0.003)),
        w_min=float(stdp_yaml.get("w_min", 0.0)),
        w_max=float(stdp_yaml.get("w_max", 1.0)),
        learning_rate=float(stdp_yaml.get("learning_rate", 0.01)),
    )
    reservoir = SequentialSNNReservoir(
        in_features=encoder.out_features,
        geometry=geom,
        lif_beta=float(res_cfg.get("lif_beta", 0.9)),
        lif_threshold=float(res_cfg.get("lif_threshold", 1.0)),
        cheb_radius=int(res_cfg.get("cheb_radius", 1)),
        long_range_edges=int(res_cfg.get("long_range_edges", 2)),
        weight_scale=float(res_cfg.get("weight_scale", 0.35)),
        stdp_cfg=stdp_cfg,
        seed=int(cfg.get("reservoir_seed", cfg.get("seed", 42))),
    ).to(device)

    head_mode = str(cfg.get("classifier_mode", "desnn"))
    head = build_output_head(
        head_mode,
        reservoir_dim=geom.n,
        num_classes=num_classes,
        desnn_hidden=cfg.get("desnn_hidden", 64),
        desnn_temperature=float(cfg.get("desnn_temperature", 1.0)),
        desnn_dropout=float(cfg.get("desnn_dropout", 0.0)),
        probe_embedding_batchnorm=bool(cfg.get("probe_embedding_batchnorm", False)),
    ).to(device)

    # --- sanity smoke ---
    x0 = torch.randn(2, window_samples, eff_c, device=device)
    sp0 = encoder(x0)
    t_sp = window_samples - 1
    _assert_spike_shape(sp0, 2, t_sp, encoder.out_features)
    reservoir.set_stdp_enabled(True)
    f0, _ = reservoir(sp0, stdp=False)
    assert f0.shape == (2, geom.n)
    reservoir.set_stdp_enabled(False)
    logits0 = head(f0)
    assert logits0.shape == (2, num_classes)
    assert_label_range(torch.tensor([0, num_classes - 1]), num_classes)
    logger.info("Sanity smoke OK (encode + reservoir + head).")

    encoder_cfg = {
        "num_channels": eff_c,
        "threshold": encoder.threshold,
        "threshold_scale": encoder.threshold_scale,
        "encoding": encoder.encoding,
    }
    reservoir_cfg = {
        "nx": geom.nx,
        "ny": geom.ny,
        "nz": geom.nz,
        "lif_beta": reservoir.lif_beta,
        "lif_threshold": reservoir.lif_threshold,
        "cheb_radius": int(res_cfg.get("cheb_radius", 1)),
        "long_range_edges": int(res_cfg.get("long_range_edges", 2)),
        "weight_scale": float(res_cfg.get("weight_scale", 0.35)),
        "in_features": encoder.out_features,
        "reservoir_seed": int(cfg.get("reservoir_seed", cfg.get("seed", 42))),
    }
    stdp_cfg_dict = stdp_cfg.__dict__.copy()
    head_cfg = {
        "mode": head_mode,
        "desnn_hidden": cfg.get("desnn_hidden", 64),
        "desnn_temperature": float(cfg.get("desnn_temperature", 1.0)),
        "desnn_dropout": float(cfg.get("desnn_dropout", 0.0)),
        "probe_embedding_batchnorm": bool(cfg.get("probe_embedding_batchnorm", False)),
    }

    # --- A–C: unsupervised STDP on train subjects ---
    stdp_epochs = int(cfg.get("stdp_epochs", 1))
    reservoir.set_stdp_enabled(True)
    reservoir.train()
    logger.info("STDP unsupervised phase: epochs=%d | train windows=%d", stdp_epochs, len(train_ds))
    for ep in range(1, stdp_epochs + 1):
        n_seen = 0
        for x, _y, _w in train_loader:
            x = _slice_batch_x(x.to(device), cache_channels=cache_channels, use_gyro=use_gyro)
            for bi in range(x.shape[0]):
                xs = x[bi : bi + 1]
                spikes = encoder(xs)
                _assert_spike_shape(spikes, 1, window_samples - 1, encoder.out_features)
                reservoir(spikes, stdp=True)
                n_seen += 1
        logger.info("STDP epoch %d/%d finished (presented %d train windows).", ep, stdp_epochs, n_seen)

    # --- D–E: freeze reservoir (no STDP; no weight updates) ---
    reservoir.set_stdp_enabled(False)
    for p in reservoir.parameters():
        p.requires_grad_(False)
    reservoir.eval()
    logger.info("Reservoir frozen after STDP (requires_grad=False, stdp disabled).")

    weight: torch.Tensor | None = None
    if bool(cfg.get("class_balanced_loss", False)):
        class_counts_t = torch.from_numpy(train_ds.count_labels(num_classes))
        weight = balanced_class_weights(class_counts_t).to(device)
        logger.info(
            "Class-balanced classifier loss: train window counts min/median/max = %d / %.0f / %d",
            int(class_counts_t.min()),
            float(class_counts_t.float().median()),
            int(class_counts_t.max()),
        )
    crit = nn.CrossEntropyLoss(
        weight=weight,
        label_smoothing=float(cfg.get("label_smoothing", 0.0)),
    )
    opt = torch.optim.AdamW(
        head.parameters(),
        lr=float(cfg.get("classifier_lr", 1e-3)),
        weight_decay=float(cfg.get("classifier_weight_decay", 1e-4)),
    )
    clf_epochs = int(cfg.get("classifier_epochs", 40))
    sched_probe_cfg = {
        "lr_scheduler": str(cfg.get("classifier_lr_scheduler", cfg.get("lr_scheduler", "none"))),
        "lr": float(cfg.get("classifier_lr", 1e-3)),
        "lr_min_ratio": float(cfg.get("classifier_lr_min_ratio", cfg.get("lr_min_ratio", 0.05))),
    }
    scheduler = build_probe_lr_scheduler(opt, sched_probe_cfg, clf_epochs)
    if scheduler is not None:
        logger.info("Classifier LR scheduler: %s", sched_probe_cfg["lr_scheduler"])
    mixup_alpha = float(cfg.get("feature_mixup_alpha", 0.0))
    if mixup_alpha > 0.0:
        logger.info("Feature mixup enabled: Beta(%s, %s) on reservoir rates", mixup_alpha, mixup_alpha)
    max_batches_dbg = cfg.get("debug_max_batches_per_epoch")
    max_batches_dbg = int(max_batches_dbg) if max_batches_dbg is not None else None

    curves: dict[str, list[float]] = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_acc = -1.0
    best_val_loss = float("inf")

    for epoch in range(1, clf_epochs + 1):
        tr_loss, tr_acc = _epoch_classifier(
            encoder,
            reservoir,
            head,
            train_loader_clf,
            device,
            crit,
            opt,
            num_classes,
            train=True,
            cache_channels=cache_channels,
            use_gyro=use_gyro,
            max_batches=max_batches_dbg,
            feature_mixup_alpha=mixup_alpha,
        )
        va_loss, va_acc = _epoch_classifier(
            encoder,
            reservoir,
            head,
            val_loader,
            device,
            crit,
            opt,
            num_classes,
            train=False,
            cache_channels=cache_channels,
            use_gyro=use_gyro,
            max_batches=max_batches_dbg,
        )
        curves["train_loss"].append(tr_loss)
        curves["val_loss"].append(va_loss)
        curves["train_acc"].append(tr_acc)
        curves["val_acc"].append(va_acc)
        logger.info(
            "clf epoch=%d train_loss=%.5f val_loss=%.5f train_acc=%.4f val_acc=%.4f",
            epoch,
            tr_loss,
            va_loss,
            tr_acc,
            va_acc,
        )
        if scheduler is not None:
            scheduler.step()

        last_payload = _payload_from_parts(
            encoder_cfg=encoder_cfg,
            reservoir_cfg=reservoir_cfg,
            stdp_cfg=stdp_cfg_dict,
            head_mode=head_mode,
            head_cfg=head_cfg,
            encoder=encoder,
            reservoir=reservoir,
            head=head,
            num_classes=num_classes,
            meta={"stage": "classifier", "epoch": epoch},
        )
        save_checkpoint(ckpt_dir / "last.pt", last_payload)

        if va_acc > best_val_acc:
            best_val_acc = float(va_acc)
            save_checkpoint(ckpt_dir / "best_val_acc.pt", last_payload)
        if va_loss < best_val_loss:
            best_val_loss = float(va_loss)
            save_checkpoint(ckpt_dir / "best_val_loss.pt", last_payload)

    write_json(ckpt_dir / "curves.json", curves)
    _save_training_curves(curves, plots_dir / "training_curves.png", str(cfg.get("experiment_name", "sequential_snn")))

    # --- quick val metrics + confusion for monitoring ---
    head.eval()
    ys: list[int] = []
    ps: list[int] = []
    with torch.no_grad():
        for x, y, _w in val_loader:
            x = _slice_batch_x(x.to(device), cache_channels=cache_channels, use_gyro=use_gyro)
            spikes = encoder(x)
            feats, _ = reservoir(spikes, stdp=False)
            pred = torch.argmax(head(feats), dim=-1)
            ys.extend(y.cpu().numpy().tolist())
            ps.extend(pred.cpu().numpy().tolist())

    labels = list(range(num_classes))
    metrics = compute_metrics(np.asarray(ys, dtype=np.int64), np.asarray(ps, dtype=np.int64), labels=labels)
    write_json(ckpt_dir / "metrics_val.json", metrics)
    index_to_code: list[str] = list(label_map["index_to_code"])
    class_names = [f"{i}:{index_to_code[i]}" for i in range(num_classes)]
    cm = np.asarray(metrics["confusion_matrix"], dtype=np.int64)
    save_confusion_matrix_figure(cm, class_names, ckpt_dir / "confusion_matrix_val.png", title="Val confusion (sequential SNN)")

    meta_final = {
        "experiment_name": str(cfg.get("experiment_name", "sequential_snn_wisdm")),
        "train_subjects": train_ids,
        "val_subjects": val_ids,
        "test_subjects": test_ids,
        "cache_channels": cache_channels,
        "effective_channels": eff_c,
        "use_gyro": use_gyro,
    }
    save_checkpoint(
        ckpt_dir / "last.pt",
        _payload_from_parts(
            encoder_cfg=encoder_cfg,
            reservoir_cfg=reservoir_cfg,
            stdp_cfg=stdp_cfg_dict,
            head_mode=head_mode,
            head_cfg=head_cfg,
            encoder=encoder,
            reservoir=reservoir,
            head=head,
            num_classes=num_classes,
            meta=meta_final,
        ),
    )
    best_path = (ckpt_dir / "best_val_acc.pt").resolve()
    if best_path.exists():
        (ckpt_dir / "eval_checkpoint.txt").write_text(str(best_path) + "\n", encoding="utf-8")
    logger.info("Training complete. Checkpoints under %s", ckpt_dir)


if __name__ == "__main__":
    main()
