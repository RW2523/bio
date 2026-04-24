#!/usr/bin/env python3
"""
Drive experiment cycle: discover artifacts, summarize Case1/Case2 vs baseline,
optionally plot SSL curves, run Case1 ablations + Case2 val-acc ablation,
embedding compare (subprocesses to existing scripts).

Usage (repo root, with venv that has torch, sklearn, matplotlib, umap-learn):
  python verify/experiment_cycle_driver.py [--repo ROOT] [--dry-run]

  --dry-run: only discover + summarize + SSL plot + recommend; skip train/eval subprocesses
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def discover(repo: Path) -> dict:
    out = {"repo": str(repo), "checkpoints": [], "metrics": [], "curves": [], "confusion": []}
    if not (repo / "outputs").is_dir():
        return out
    for p in repo.glob("outputs/**/*.pt"):
        out["checkpoints"].append(str(p.resolve()))
    for p in repo.glob("outputs/**/metrics.json"):
        out["metrics"].append(str(p.resolve()))
    for p in repo.glob("outputs/**/curves.json"):
        out["curves"].append(str(p.resolve()))
    for p in repo.glob("outputs/**/confusion_matrix.png"):
        out["confusion"].append(str(p.resolve()))
    return out


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def guess_eval_for_ckpt(repo: Path, ckpt_name: str) -> Path | None:
    """Heuristic: outputs/eval_runs/<stem>/metrics.json"""
    stem = Path(ckpt_name).parent.name
    cand = repo / "outputs" / "eval_runs" / stem / "metrics.json"
    if cand.is_file():
        return cand
    return None


def num_classes_from_artifacts(repo: Path) -> int | None:
    p = repo / "outputs" / "artifacts" / "label_map.json"
    if not p.is_file():
        return None
    return int(load_json(p)["num_classes"])


def summarize_case_metrics(repo: Path, case: str) -> dict | None:
    ckpt = repo / "outputs" / "checkpoints" / case / "best.pt"
    if not ckpt.is_file():
        return None
    metrics_path = guess_eval_for_ckpt(repo, str(ckpt))
    if metrics_path is None or not metrics_path.is_file():
        return {"checkpoint": str(ckpt), "metrics_path": None, "missing_eval": True}
    m = load_json(metrics_path)
    return {
        "checkpoint": str(ckpt.resolve()),
        "metrics_path": str(metrics_path.resolve()),
        "accuracy": m.get("accuracy"),
        "macro_f1": m.get("macro_f1"),
        "weighted_f1": m.get("weighted_f1"),
        "cohen_kappa": m.get("cohen_kappa"),
    }


def plot_ssl_curves(repo: Path, out_png: Path) -> bool:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    curves_p = repo / "outputs" / "checkpoints" / "ssl" / "curves.json"
    if not curves_p.is_file():
        return False
    c = load_json(curves_p)
    tr, va = c.get("train_loss", []), c.get("val_loss", [])
    if not tr:
        return False
    ep = list(range(1, len(tr) + 1))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(ep, tr, label="train_loss")
    ax.plot(ep, va, label="val_loss")
    ax.set_xlabel("epoch")
    ax.set_ylabel("SSL loss (sum of 3 BCE)")
    ax.legend()
    ax.set_title("SSL pretraining (AugPred)")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return True


def run_cmd(cmd: list[str], repo: Path) -> int:
    print("+", " ".join(cmd))
    return subprocess.call(cmd, cwd=str(repo))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument("--dry-run", action="store_true", help="Skip training/eval subprocesses")
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()
    repo: Path = args.repo.resolve()

    disc = discover(repo)
    print("=== 1) DISCOVER ===")
    print(json.dumps({k: v for k, v in disc.items() if k != "checkpoints"}, indent=2))
    print(f"checkpoints count: {len(disc['checkpoints'])}")

    if not (repo / "outputs").is_dir():
        print("\nBLOCKED: no outputs/ directory. Run preprocess + train + eval on this machine, then re-run this driver.")
        sys.exit(2)

    nc = num_classes_from_artifacts(repo)
    baseline = 1.0 / nc if nc else None

    print("\n=== 2) SUMMARY TABLE (Case1 / Case2) ===")
    rows = []
    for case in ("case1", "case2"):
        s = summarize_case_metrics(repo, case)
        if s and not s.get("missing_eval"):
            rows.append(
                {
                    "case": case,
                    "test_acc": s["accuracy"],
                    "macro_f1": s["macro_f1"],
                    "weighted_f1": s["weighted_f1"],
                    "kappa": s["cohen_kappa"],
                    "checkpoint": s["checkpoint"],
                    "random_baseline_1/C": baseline,
                }
            )
            print(json.dumps(rows[-1], indent=2))
        elif s:
            print(case, "checkpoint exists but no matching eval metrics.json — run eval first.")
        else:
            print(case, "no checkpoint at outputs/checkpoints/", case, "/best.pt")

    print("\n=== 3) CASE2 vs CASE1 ===")
    if len(rows) >= 2:
        a1 = next(r for r in rows if r["case"] == "case1")
        a2 = next(r for r in rows if r["case"] == "case2")
        d = float(a2["test_acc"]) - float(a1["test_acc"])
        print(f"Δ test accuracy (Case2 - Case1): {d:+.4f}")
        if abs(d) < 0.01:
            print("Gap: tiny (<1 point)")
        elif abs(d) < 0.03:
            print("Gap: small")
        else:
            print("Gap: meaningful")
        print("Case2 better than Case1:", d > 0)
    else:
        print("Insufficient rows to compare.")

    ssl_plot = repo / "outputs" / "features" / "ssl_loss_curves.png"
    if plot_ssl_curves(repo, ssl_plot):
        print("\n=== 7) SSL plot written ===", ssl_plot)
        c = load_json(repo / "outputs" / "checkpoints" / "ssl" / "curves.json")
        ne = len(c.get("train_loss", []))
        print(f"SSL epochs in curves: {ne}")
        if ne <= 5:
            print("5 epochs or fewer on disk — often short for representation learning. Try 20–50 next if val_loss still decreasing.")
    else:
        print("\n=== 7) No SSL curves.json ===")

    if args.dry_run:
        print("\n--dry-run: skipping ablations 4–6 and embedding step 8.")
        sys.exit(0)

    py = sys.executable
    art = repo / "outputs" / "artifacts"
    if not art.is_dir():
        print("Missing outputs/artifacts; cannot run eval/ablations.")
        sys.exit(3)

    ablations = [
        ("case1_ablation_best_val_acc", "case1_ablation_best_val_acc"),
        ("case1_ablation_no_class_balance", "case1_ablation_no_class_balance"),
        ("case1_ablation_no_label_smooth", "case1_ablation_no_label_smooth"),
        ("case1_ablation_no_weight_decay", "case1_ablation_no_weight_decay"),
    ]
    print("\n=== 4) Case1 ablations (train + eval) ===")
    for name, cfg in ablations:
        if run_cmd([py, "train/train_linear_probe_case1.py", "--config", cfg], repo) != 0:
            print("FAILED train", cfg)
            continue
        ckpt = repo / "outputs" / "checkpoints" / name / "best.pt"
        evdir = repo / "outputs" / "eval_runs" / name
        run_cmd(
            [
                py,
                "eval/evaluate.py",
                "--checkpoint",
                str(ckpt),
                "--artifacts_dir",
                str(art),
                "--output_dir",
                str(evdir),
                "--device",
                args.device,
            ],
            repo,
        )

    print("\n=== 5) Case2 best-val-acc ablation ===")
    case2_cfg = repo / "configs" / "case2_ablation_best_val_acc.yaml"
    if not case2_cfg.is_file():
        print("Missing configs/case2_ablation_best_val_acc.yaml — create it (inherits case2_probe + best_checkpoint_metric + ckpt dir).")
    else:
        run_cmd([py, "train/train_linear_probe_case2.py", "--config", "case2_ablation_best_val_acc"], repo)
        ckpt = repo / "outputs" / "checkpoints" / "case2_ablation_best_val_acc" / "best.pt"
        run_cmd(
            [
                py,
                "eval/evaluate.py",
                "--checkpoint",
                str(ckpt),
                "--artifacts_dir",
                str(art),
                "--output_dir",
                str(repo / "outputs" / "eval_runs" / "case2_ablation_best_val_acc"),
                "--device",
                args.device,
            ],
            repo,
        )

    print("\n=== 6) Best single ablation (max Case1 test acc) ===")
    best_name, best_acc = None, -1.0
    for name, _ in ablations:
        mp = repo / "outputs" / "eval_runs" / name / "metrics.json"
        if mp.is_file():
            acc = float(load_json(mp)["accuracy"])
            if acc > best_acc:
                best_acc, best_name = acc, name
    print("winner:", best_name, "acc=", best_acc)

    print("\n=== 8) Embeddings random vs SSL ===")
    c1 = repo / "outputs" / "checkpoints" / "case1" / "best.pt"
    sslb = repo / "outputs" / "checkpoints" / "ssl" / "backbone_best.pt"
    if c1.is_file() and sslb.is_file():
        fe = repo / "outputs" / "features"
        fe.mkdir(parents=True, exist_ok=True)
        rnp = fe / "emb_train_random.npz"
        snp = fe / "emb_train_ssl.npz"
        run_cmd(
            [
                py,
                "eval/extract_features.py",
                "--backbone_checkpoint",
                str(c1),
                "--split",
                "train",
                "--artifacts_dir",
                str(art),
                "--output",
                str(rnp),
                "--device",
                args.device,
            ],
            repo,
        )
        run_cmd(
            [
                py,
                "eval/extract_features.py",
                "--backbone_checkpoint",
                str(sslb),
                "--split",
                "train",
                "--artifacts_dir",
                str(art),
                "--output",
                str(snp),
                "--device",
                args.device,
            ],
            repo,
        )
        run_cmd(
            [
                py,
                "eval/compare_embedding_backbones.py",
                "--embeddings_random",
                str(rnp),
                "--embeddings_pretrained",
                str(snp),
                "--artifacts_dir",
                str(art),
                "--output",
                str(fe / "compare_train_umap.png"),
                "--method",
                "umap",
            ],
            repo,
        )
    else:
        print("Skip embeddings: need case1/best.pt and ssl/backbone_best.pt")

    print("\n=== DONE ===")


if __name__ == "__main__":
    main()
