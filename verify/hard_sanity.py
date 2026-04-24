#!/usr/bin/env python3
"""
Hard sanity checks (runs training on synthetic data matching pipeline shapes).

Run from repo root:
  . .venv_sanity/bin/activate   # or any env with torch
  python verify/hard_sanity.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
from collections import Counter

from sklearn.metrics import confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.linear_probe import LinearProbeHead
from models.spiking_resnet1d import SpikingResNet1d
from train.common import freeze_module, to_bct


def _tiny_backbone(device: torch.device) -> SpikingResNet1d:
    return SpikingResNet1d(
        in_channels=3,
        base_channels=16,
        layers=(1, 1),
        stem_kernel=5,
        beta=0.9,
        threshold=1.0,
        surrogate_alpha=2.0,
        reset="subtract",
    ).to(device)


def _synthetic_batch(*, n: int, t: int, num_classes: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """x [n,t,3] like dataloader; y [n] long."""
    x = torch.randn(n, t, 3, device=device)
    y = torch.randint(0, num_classes, (n,), device=device)
    return x, y


def check_labels_logits_loss(*, num_classes: int = 8, device: torch.device) -> dict:
    """Checks 6–8: alignment, index range, CE expects raw logits."""
    n, t = 16, 64
    x, y = _synthetic_batch(n=n, t=t, num_classes=num_classes, device=device)
    assert y.dtype == torch.long
    assert int(y.min()) >= 0 and int(y.max()) < num_classes

    bb = _tiny_backbone(device)
    head = LinearProbeHead(bb.out_dim, num_classes, embedding_batchnorm=False).to(device)
    logits = head(bb(to_bct(x)))
    assert logits.shape == (n, num_classes)
    crit = nn.CrossEntropyLoss()
    loss = crit(logits, y)
    loss.backward()

    # CE must match NLL(softmax(logits), y) is WRONG; CE expects logits — compare to manual log_softmax
    with torch.no_grad():
        ce2 = nn.functional.cross_entropy(logits, y)
        assert torch.allclose(loss.detach(), ce2)

    return {"logits_shape": list(logits.shape), "y_range": [int(y.min()), int(y.max())], "ce_ok": True}


def overfit_experiment(
    *,
    n_samples: int,
    freeze_backbone: bool,
    steps: int,
    device: torch.device,
) -> dict:
    torch.manual_seed(0)
    num_classes = 6
    t = 64
    x_all, y_all = _synthetic_batch(n=n_samples, t=t, num_classes=num_classes, device=device)

    backbone = _tiny_backbone(device)
    head = LinearProbeHead(backbone.out_dim, num_classes, embedding_batchnorm=True).to(device)

    if freeze_backbone:
        freeze_module(backbone)

    params = list(head.parameters()) if freeze_backbone else list(backbone.parameters()) + list(head.parameters())
    opt = torch.optim.AdamW(params, lr=3e-3 if freeze_backbone else 1e-3, weight_decay=0.0)
    crit = nn.CrossEntropyLoss()

    acc_hist: list[float] = []
    loss_hist: list[float] = []

    for step in range(steps):
        backbone.eval() if freeze_backbone else backbone.train()
        head.train()
        opt.zero_grad(set_to_none=True)
        if freeze_backbone:
            with torch.no_grad():
                z = backbone(to_bct(x_all))
        else:
            z = backbone(to_bct(x_all))
        logits = head(z)
        loss = crit(logits, y_all)
        loss.backward()
        opt.step()

        with torch.no_grad():
            pred = logits.argmax(dim=-1)
            acc = float((pred == y_all).float().mean().item())
        acc_hist.append(acc)
        loss_hist.append(float(loss.item()))

    with torch.no_grad():
        z = backbone(to_bct(x_all))
        logits = head(z)
        pred = logits.argmax(dim=-1).cpu().numpy()
        y_np = y_all.cpu().numpy()
        cm = confusion_matrix(y_np, pred, labels=list(range(num_classes)))

    return {
        "freeze_backbone": freeze_backbone,
        "n_samples": n_samples,
        "steps": steps,
        "final_train_acc": acc_hist[-1],
        "max_train_acc": max(acc_hist),
        "first_acc": acc_hist[0],
        "final_loss": loss_hist[-1],
        "first_loss": loss_hist[0],
        "acc_rising": acc_hist[-1] > acc_hist[0],
        "loss_falling": loss_hist[-1] < loss_hist[0],
        "pred_unique_classes": int(len(set(pred.tolist()))),
        "cm": cm.tolist(),
        "dominant_pred_frac": float((pred == int(pred[0])).mean()) if len(pred) else 0.0,
    }


def overfit_probe_after_warmup(*, n_samples: int, device: torch.device) -> dict:
    """Joint train backbone+head briefly, freeze backbone, then train head — probe should memorize."""
    torch.manual_seed(2)
    num_classes = 6
    t = 64
    x_all, y_all = _synthetic_batch(n=n_samples, t=t, num_classes=num_classes, device=device)
    backbone = _tiny_backbone(device)
    head = LinearProbeHead(backbone.out_dim, num_classes, embedding_batchnorm=True).to(device)
    opt = torch.optim.AdamW(list(backbone.parameters()) + list(head.parameters()), lr=3e-3)
    crit = nn.CrossEntropyLoss()
    for _ in range(200):
        backbone.train()
        head.train()
        opt.zero_grad(set_to_none=True)
        logits = head(backbone(to_bct(x_all)))
        crit(logits, y_all).backward()
        opt.step()
    freeze_module(backbone)
    opt = torch.optim.AdamW(head.parameters(), lr=1e-2)
    for _ in range(500):
        backbone.eval()
        head.train()
        opt.zero_grad(set_to_none=True)
        with torch.no_grad():
            z = backbone(to_bct(x_all))
        logits = head(z)
        crit(logits, y_all).backward()
        opt.step()
    with torch.no_grad():
        acc = float((head(backbone(to_bct(x_all))).argmax(-1) == y_all).float().mean())
    return {"n_samples": n_samples, "final_train_acc_after_warmup_freeze": acc}


def collapse_probe(*, device: torch.device) -> dict:
    """Train on 95% class 0 — see if predictions collapse."""
    torch.manual_seed(1)
    n, t, num_classes = 80, 64, 5
    x = torch.randn(n, t, 3, device=device)
    y = torch.zeros(n, dtype=torch.long, device=device)
    y[: int(0.05 * n)] = torch.randint(1, num_classes, (int(0.05 * n),), device=device)

    backbone = _tiny_backbone(device)
    freeze_module(backbone)
    head = LinearProbeHead(backbone.out_dim, num_classes, embedding_batchnorm=False).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=1e-2)
    crit = nn.CrossEntropyLoss()

    for _ in range(400):
        opt.zero_grad(set_to_none=True)
        with torch.no_grad():
            z = backbone(to_bct(x))
        logits = head(z)
        crit(logits, y).backward()
        opt.step()

    with torch.no_grad():
        pred = head(backbone(to_bct(x))).argmax(dim=-1).cpu().numpy()
    counts = {int(c): int((pred == c).sum()) for c in range(num_classes)}
    maj = int(Counter(pred.tolist()).most_common(1)[0][0])
    frac_maj = float((pred == maj).mean())
    return {"pred_hist": counts, "majority_pred_class": maj, "frac_predicted_majority": frac_maj}


def main() -> None:
    device = torch.device("cpu")
    print("=== hard_sanity.py (synthetic data, repo SpikingResNet1d + LinearProbeHead) ===\n")

    r = check_labels_logits_loss(device=device)
    print("6–8. Labels / logits / CE:", r)

    print("\n1. Overfit ~40 samples (backbone + head trainable, same batch every step)")
    r1 = overfit_experiment(n_samples=40, freeze_backbone=False, steps=300, device=device)
    print(r1)
    ok1 = r1["final_train_acc"] >= 0.95 or r1["max_train_acc"] >= 0.95
    print(f"   PASS overfit full model: {ok1}\n")

    print("2a. Overfit ~40 samples (frozen random backbone from init, train head only)")
    r2 = overfit_experiment(n_samples=40, freeze_backbone=True, steps=800, device=device)
    print(r2)
    ok2 = r2["max_train_acc"] >= 0.95
    print(f"   PASS probe-only from random init: {ok2} (expected False: labels not linearly sep. in fixed random feats)\n")

    print("2b. Same 40 samples: joint warmup 200 steps, freeze backbone, train head 500 steps")
    r2b = overfit_probe_after_warmup(n_samples=40, device=device)
    print(r2b)
    ok2b = r2b["final_train_acc_after_warmup_freeze"] >= 0.95
    print(f"   PASS probe after warmup freeze: {ok2b}\n")

    print("3–5. Acc trend + pred diversity on balanced overfit (check 1 again)")
    print(f"   acc_rising={r1['acc_rising']}, loss_falling={r1['loss_falling']}")
    print(f"   pred_unique_classes={r1['pred_unique_classes']} / 6 (should use multiple classes if acc high)")
    print(f"   confusion_matrix:\n{r1['cm']}\n")

    print("4–5 (imbalanced). After training on 95% class 0:")
    c = collapse_probe(device=device)
    print(c)

    # Curves.json inspection if present
    curves_path = ROOT / "outputs" / "checkpoints" / "case1" / "curves.json"
    print("\n--- Real run curves (if exists):", curves_path)
    if curves_path.is_file():
        import json

        data = json.loads(curves_path.read_text())
        tr = data.get("train_loss", [])
        va = data.get("val_acc", [])
        if va:
            print(f"   val_acc first/last: {va[0]:.4f} -> {va[-1]:.4f}, rising={va[-1] > va[0]}")
        if tr:
            print(f"   train_loss first/last: {tr[0]:.5f} -> {tr[-1]:.5f}, falling={tr[-1] < tr[0]}")
    else:
        print("   (no curves.json — skipped)")

    # Summary verdict
    print("\n=== SUMMARY ===")
    fails = []
    if not ok1:
        fails.append("full_model_overfit")
    if not ok2b:
        fails.append("probe_after_warmup_freeze")
    if not r1["acc_rising"]:
        fails.append("acc_not_rising")
    if r1["final_train_acc"] >= 0.9 and r1["pred_unique_classes"] < 2:
        fails.append("pred_collapse_despite_high_acc")

    print("first_fail:", fails[0] if fails else "none")
    if fails:
        print(
            "interpretation: If probe_after_warmup_freeze fails, head+freeze path is broken. "
            "If full_model_overfit fails, optimizer/backward path is broken. "
            "probe-only from random init (2a) failing is expected (non-separable fixed features)."
        )


if __name__ == "__main__":
    main()
