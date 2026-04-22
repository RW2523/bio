#!/bin/bash
# =============================================================================
# UMass Unity Slurm — COMPLETE Case 2 pipeline (preprocess + SSL + Case 2 + eval)
#
# Mirrors cluster/unity_case1_full_a100.sh, but runs the AugPred SSL path first,
# then trains the frozen SSL-backbone linear probe (Case 2), then evaluates.
#
# Steps on one GPU node (A100):
#   1) Dataset audit
#   2) Manifest
#   3) Full preprocess (all subjects after data.yaml patch)
#   4) AugPred SSL pretrain (configs/ssl.yaml) → outputs/checkpoints/ssl/backbone_best.pt
#   5) Case 2 linear probe (configs/case2_probe.yaml; loads pretrained backbone)
#   6) Held-out test evaluation → outputs/eval_runs/case2/
#
# Does NOT run: Case 1.
#
# Unity references:
#   https://docs.unity.rc.umass.edu/documentation/tools/gpus/
#   https://docs.unity.rc.umass.edu/documentation/jobs/sbatch
#
# ------------------------------------------------------------------------------
# BEFORE FIRST RUN (login node)
#
#   1) Clone repo, venv, pip install -r requirements.txt (CUDA PyTorch on cluster)
#   2) WISDM extract: raw/.../activity_key.txt
#   3) Submit from repo root (recommended):
#        cd /path/to/bio
#        export WISDM_DATA_ROOT=/path/to/wisdm-dataset   # optional if default sibling path is correct
#        sbatch cluster/unity_case2_full_a100.sh
#
#   From another directory:
#        export PROJECT_ROOT=/path/to/bio
#        export WISDM_DATA_ROOT=/path/to/wisdm-dataset
#        sbatch --chdir="$PROJECT_ROOT" "$PROJECT_ROOT/cluster/unity_case2_full_a100.sh"
#
#   Case 2 needs longer than Case 1 (SSL + probe). If Unity rejects walltime, override:
#        sbatch --time=3-00:00:00 --mem=256G cluster/unity_case2_full_a100.sh
#
# ------------------------------------------------------------------------------
# OUTPUTS (under ${PROJECT_ROOT}/outputs/)
#
#   outputs/audit/                    — audit + manifest
#   outputs/artifacts/                — splits, norm_stats, label_map, preprocess_run.json
#   outputs/cache/wisdm_windows/      — window .npz cache
#   outputs/checkpoints/ssl/          — SSL checkpoints + backbone_best.pt
#   outputs/checkpoints/case2/        — Case 2 best.pt, last.pt, train.log, curves.json
#   outputs/eval_runs/case2/          — metrics, classification_report, confusion_matrix.png
#   outputs/logs/slurm-case2-full-* — Slurm stdout/stderr (relative to submit cwd; use cd repo)
# =============================================================================

#SBATCH --job-name=wisdm-case2-full
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00
#SBATCH --gpus=1
#SBATCH --constraint=a100
#SBATCH --output=outputs/logs/slurm-case2-full-%j.out
#SBATCH --error=outputs/logs/slurm-case2-full-%j.err

set -euo pipefail

# ------------- Paths (same pattern as unity_case1_full_a100.sh) ------------
: "${PROJECT_ROOT:=${SLURM_SUBMIT_DIR:-}}"
DEFAULT_WISDM_DATA_ROOT="$(dirname "${PROJECT_ROOT}")/wisdm-dataset"
: "${WISDM_DATA_ROOT:=${DEFAULT_WISDM_DATA_ROOT}}"

if [[ "${WISDM_DATA_ROOT}" == "/absolute/path/to/wisdm-dataset" ]]; then
  echo "ERROR: WISDM_DATA_ROOT is still the placeholder path /absolute/path/to/wisdm-dataset." >&2
  echo "       Use ${DEFAULT_WISDM_DATA_ROOT} for this repo layout, or pass your real dataset path." >&2
  exit 1
fi

if [[ ! -f "${PROJECT_ROOT}/train/train_linear_probe_case2.py" ]]; then
  echo "ERROR: PROJECT_ROOT must be the repository root. Current: ${PROJECT_ROOT}" >&2
  echo "       cd into the repo before sbatch, or: export PROJECT_ROOT=/absolute/path/to/repo" >&2
  exit 1
fi

if [[ ! -d "${WISDM_DATA_ROOT}" ]]; then
  echo "ERROR: WISDM_DATA_ROOT=${WISDM_DATA_ROOT} does not exist." >&2
  echo "       Default for this layout: ${DEFAULT_WISDM_DATA_ROOT}" >&2
  exit 1
fi

if [[ ! -d "${WISDM_DATA_ROOT}/raw" || ! -f "${WISDM_DATA_ROOT}/activity_key.txt" ]]; then
  echo "ERROR: WISDM_DATA_ROOT=${WISDM_DATA_ROOT} is missing raw/ or activity_key.txt." >&2
  exit 1
fi

: "${VENV_ROOT:=${PROJECT_ROOT}/.venv}"
export PROJECT_ROOT WISDM_DATA_ROOT VENV_ROOT

if [[ ! -f "${VENV_ROOT}/bin/activate" ]]; then
  echo "ERROR: Missing venv at ${VENV_ROOT}" >&2
  echo "       Create with: python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt" >&2
  exit 1
fi

cd "${PROJECT_ROOT}"
mkdir -p outputs/audit outputs/artifacts outputs/logs \
  outputs/eval_runs/case2 outputs/checkpoints/case2 outputs/checkpoints/ssl

# shellcheck source=/dev/null
source "${VENV_ROOT}/bin/activate"

# Optional: match system CUDA/cuDNN to your torch build (module spider cuda)
# module load cuda/12.6
# module load cudnn/8.9.7.29-12-cuda12.6

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"

echo "=== Job (Case 2 full) ==="
echo "PROJECT_ROOT=${PROJECT_ROOT}"
echo "WISDM_DATA_ROOT=${WISDM_DATA_ROOT}"
echo "HOST=$(hostname) SLURM_JOB_ID=${SLURM_JOB_ID:-local}"
python - <<'VERS'
import sys
try:
    import torch
    print("python", sys.version.split()[0], "| torch", torch.__version__, "| cuda?", torch.cuda.is_available())
except Exception as e:
    print("python", sys.version.split()[0], "| torch import failed:", e)
    sys.exit(1)
VERS

# --- Patch configs/data.yaml: cluster path + full cohort --------------------
DATA_YAML="${PROJECT_ROOT}/configs/data.yaml"
DATA_BAK="${SLURM_TMPDIR:-/tmp}/data.yaml.bak.${SLURM_JOB_ID:-$$}"
cp -a "${DATA_YAML}" "${DATA_BAK}"
restore_data_yaml() { cp -a "${DATA_BAK}" "${DATA_YAML}"; }
trap restore_data_yaml EXIT

python3 <<'PATCH' || exit 1
import os
from pathlib import Path

import yaml

pr = Path(os.environ["PROJECT_ROOT"])
dr = Path(os.environ["WISDM_DATA_ROOT"]).resolve()
p = pr / "configs" / "data.yaml"
cfg = yaml.safe_load(p.read_text(encoding="utf-8"))
cfg["data_root"] = str(dr)
cfg["max_subjects"] = None
cfg["debug_subjects"] = []
p.write_text(yaml.dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
print(f"Patched data.yaml: data_root={dr} max_subjects=null debug_subjects=[]")
PATCH

# --- Verify merged configs --------------------------------------------------
python3 <<'VERIFY' || exit 1
import os
import sys
from pathlib import Path

pr = Path(os.environ["PROJECT_ROOT"])
sys.path.insert(0, str(pr))
os.chdir(pr)
from utils.yaml_config import load_merged_config
from train.common import resolve_path

cfg = load_merged_config("preprocess")
if cfg.get("max_subjects") is not None:
    raise SystemExit(f"Refusing: merged preprocess has max_subjects={cfg.get('max_subjects')!r}")
if cfg.get("debug_subjects"):
    raise SystemExit(f"Refusing: merged preprocess has debug_subjects={cfg.get('debug_subjects')!r}")
print("VERIFY OK: preprocess uses all discovered subjects.")

ssl = load_merged_config("ssl")
print(
    "SSL config:",
    "epochs=", ssl.get("epochs"),
    "batch_size=", ssl.get("batch_size"),
    "compute_device=", ssl.get("compute_device"),
)

c2 = load_merged_config("case2_probe")
bb = c2.get("pretrained_backbone_path")
if not bb:
    raise SystemExit("case2_probe.yaml must set pretrained_backbone_path (SSL backbone checkpoint).")
bb_path = resolve_path(str(bb))
print(
    "Case2 config:",
    "epochs=", c2.get("epochs"),
    "batch_size=", c2.get("batch_size"),
    "pretrained_backbone_path (YAML)=", bb,
    "resolved (after SSL)=", bb_path,
    "probe_optimizer=", c2.get("probe_optimizer", "sgd"),
    "lr=", c2.get("lr"),
)
if int(c2.get("epochs", 0)) < 2:
    print("WARNING: case2_probe epochs < 2 is only for debugging.", file=sys.stderr)
VERIFY

echo "=== GPU ==="
nvidia-smi -L || true

OUT_AUDIT="${PROJECT_ROOT}/outputs/audit"
ART_DIR="${PROJECT_ROOT}/outputs/artifacts"
SSL_BB="${PROJECT_ROOT}/outputs/checkpoints/ssl/backbone_best.pt"

echo "=== 1/6 Dataset audit ==="
python data_tools/inspect_dataset.py --data_root "${WISDM_DATA_ROOT}" --out_dir "${OUT_AUDIT}"

echo "=== 2/6 Manifest ==="
python data_tools/build_manifest.py --data_root "${WISDM_DATA_ROOT}" --out_dir "${OUT_AUDIT}"

echo "=== 3/6 Preprocess (full cohort; --config preprocess) ==="
python data_tools/preprocess_wisdm.py --config preprocess

echo "=== 4/6 AugPred SSL pretrain (--config ssl) ==="
python train/pretrain_augpred.py --config ssl

if [[ ! -f "${SSL_BB}" ]]; then
  echo "ERROR: SSL did not produce ${SSL_BB}" >&2
  echo "       Check outputs/checkpoints/ssl/pretrain.log and last.pt / best.pt." >&2
  exit 1
fi
echo "OK: SSL backbone checkpoint ${SSL_BB}"

echo "=== 5/6 Train Case 2 frozen SSL backbone + probe (--config case2_probe) ==="
python train/train_linear_probe_case2.py --config case2_probe

echo "=== 6/6 Evaluate held-out test (Case 2) ==="
python eval/evaluate.py \
  --checkpoint "${PROJECT_ROOT}/outputs/checkpoints/case2/best.pt" \
  --artifacts_dir "${ART_DIR}" \
  --config case2_probe \
  --output_dir "${PROJECT_ROOT}/outputs/eval_runs/case2"

echo ""
echo "=== Case 2 pipeline finished OK ==="
echo "  SSL backbone: ${SSL_BB}"
echo "  Checkpoints:  ${PROJECT_ROOT}/outputs/checkpoints/case2/"
echo "  Metrics:      ${PROJECT_ROOT}/outputs/eval_runs/case2/metrics.json"
echo "  Report:       ${PROJECT_ROOT}/outputs/eval_runs/case2/classification_report.txt"
echo "  Figure:       ${PROJECT_ROOT}/outputs/eval_runs/case2/confusion_matrix.png"
