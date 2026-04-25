#!/bin/bash
# =============================================================================
# UMass Unity (Slurm) — WISDM sequential reservoir SNN (TR + STDP + readout)
#
# Full pipeline on the GPU node:
#   1) patch data_root  2) preprocess  3) train (STDP + classifier)  4) test eval  5) analysis
#
# Resource template matches repo Unity full-GPU jobs:
#   --time=04:00:00 and --constraint=a100 are fixed here by policy; change other
#   SBATCH lines (account, partition, mem) if your site requires it.
# Full pipeline: omit --qos=short (short QoS is for small priority-boosted jobs;
# can confuse long runs). For a one-off interactive boost on a *short* test, add
# on the sbatch line: --qos=short (see Unity docs).
#
# Python: this script does NOT install packages by default. Create the venv once
# on a login node (see setup.md), then e.g. pip install -r requirements.txt
# (CUDA-matched torch if you use GPU). Optional: export INSTALL_REQUIREMENTS=1
# before sbatch to run pip install -r requirements.txt inside the job (slower).
#
# Submit from repo root (directory containing train/, configs/, sbatch/):
#   export WISDM_DATA_ROOT=/path/to/wisdm-dataset
#   export SEQUENTIAL_CONFIG=sequential_snn_wisdm   # YAML name under configs/ (no .yaml)
#   sbatch sbatch/run_wisdm_sequential_snn.sbatch
#
# Optional: export SEQUENTIAL_CHECKPOINT=/abs/path/best_val_acc.pt to skip auto path.
# Eval/analysis dirs are then inferred as <output_root>/eval_runs/... where output_root
# is two levels above .../checkpoints/sequential_snn/best_val_acc.pt.
#
# Logs: slurm-unity-seq-snn-<jobid>.{out,err} plus logs/sequential_snn/*.log
# =============================================================================

#SBATCH --job-name=wisdm-unity-seq-snn
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --gpus=1
#SBATCH --constraint=a100
#SBATCH --output=slurm-unity-seq-snn-%j.out
#SBATCH --error=slurm-unity-seq-snn-%j.err

set -euo pipefail

: "${PROJECT_ROOT:=${SLURM_SUBMIT_DIR}}"
: "${VENV_ROOT:=${PROJECT_ROOT}/.venv}"
: "${INSTALL_REQUIREMENTS:=0}"
: "${SEQUENTIAL_CONFIG:=sequential_snn_wisdm}"
: "${LOG_DIR:=${PROJECT_ROOT}/logs/sequential_snn}"

if [[ -z "${WISDM_DATA_ROOT:-}" ]]; then
  echo "ERROR: export WISDM_DATA_ROOT=/path/to/wisdm-dataset before sbatch." >&2
  exit 1
fi

cd "${PROJECT_ROOT}"
mkdir -p "${LOG_DIR}"

if [[ ! -f "train/train_wisdm_sequential_snn.py" ]]; then
  echo "ERROR: PROJECT_ROOT=${PROJECT_ROOT} is not the repo root." >&2
  exit 1
fi

if [[ ! -f "${VENV_ROOT}/bin/activate" ]]; then
  echo "ERROR: venv not found at VENV_ROOT=${VENV_ROOT}. Create it and pip install -r requirements.txt (see setup.md), or set VENV_ROOT." >&2
  exit 1
fi

# shellcheck source=/dev/null
source "${VENV_ROOT}/bin/activate"

if [[ "${INSTALL_REQUIREMENTS}" == "1" ]]; then
  echo "=== pip install -r requirements.txt (INSTALL_REQUIREMENTS=1) ==="
  python3 -m pip install --upgrade pip
  python3 -m pip install -r "${PROJECT_ROOT}/requirements.txt"
fi

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"

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
p.write_text(yaml.dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
print(f"Patched data_root -> {dr}")
PATCH

echo "=== GPU check ==="
nvidia-smi -L || true

echo "=== Preprocess (window cache + splits + norm stats) ==="
python3 data_tools/preprocess_wisdm.py --config preprocess 2>&1 | tee -a "${LOG_DIR}/preprocess-${SLURM_JOB_ID:-local}.log"

echo "=== Train sequential SNN (reservoir STDP + supervised head) ==="
python3 train/train_wisdm_sequential_snn.py --config "${SEQUENTIAL_CONFIG}" 2>&1 | tee -a "${LOG_DIR}/train-${SLURM_JOB_ID:-local}.log"

# Resolve checkpoint + eval dirs from merged YAML (supports output_dir e.g. outputs2)
if [[ -n "${SEQUENTIAL_CHECKPOINT:-}" ]]; then
  CKPT="${SEQUENTIAL_CHECKPOINT}"
  # .../<output_root>/checkpoints/sequential_snn/best_val_acc.pt -> output_root
  if [[ -f "${CKPT}" ]]; then
    ROOT_OUT="$(cd "$(dirname "${CKPT}")/../.." && pwd)"
  else
    ROOT_OUT="${PROJECT_ROOT}/outputs"
  fi
  EVAL_OUT="${ROOT_OUT}/eval_runs/sequential_snn_test"
  ANALYSIS_OUT="${ROOT_OUT}/eval_runs/sequential_snn_analysis"
  echo "Using SEQUENTIAL_CHECKPOINT=${CKPT} -> ROOT_OUT=${ROOT_OUT}"
else
  eval "$(python3 <<'RESOLVE' || exit 1
import os
import shlex
import sys
from pathlib import Path

ROOT = Path(os.environ["PROJECT_ROOT"]).resolve()
sys.path.insert(0, str(ROOT))
from train.common import resolve_path
from utils.yaml_config import load_merged_config

name = os.environ.get("SEQUENTIAL_CONFIG", "sequential_snn_wisdm")
cfg = load_merged_config(name)
out = resolve_path(cfg["output_dir"])
sub = str(cfg.get("sequential_checkpoint_subdir", "checkpoints/sequential_snn"))
ckpt = (out / sub / "best_val_acc.pt").resolve()
eval_out = (out / "eval_runs" / "sequential_snn_test").resolve()
analysis_out = (out / "eval_runs" / "sequential_snn_analysis").resolve()
print(f"CKPT={shlex.quote(str(ckpt))}")
print(f"EVAL_OUT={shlex.quote(str(eval_out))}")
print(f"ANALYSIS_OUT={shlex.quote(str(analysis_out))}")
RESOLVE
)"
fi

if [[ ! -f "${CKPT}" ]]; then
  echo "ERROR: checkpoint not found after training: ${CKPT}" >&2
  echo "Check train log and SEQUENTIAL_CONFIG / output_dir / sequential_checkpoint_subdir in YAML." >&2
  exit 1
fi

mkdir -p "${EVAL_OUT}" "${ANALYSIS_OUT}"

echo "=== Evaluate test split (full trained model) ==="
python3 eval/evaluate_sequential_snn.py \
  --checkpoint "${CKPT}" \
  --output_dir "${EVAL_OUT}" \
  --device cuda 2>&1 | tee -a "${LOG_DIR}/eval-${SLURM_JOB_ID:-local}.log"

echo "=== Interaction analysis (optional explainability) ==="
python3 eval/analyze_sequential_snn.py \
  --checkpoint "${CKPT}" \
  --output_dir "${ANALYSIS_OUT}" \
  --device cuda 2>&1 | tee -a "${LOG_DIR}/analyze-${SLURM_JOB_ID:-local}.log"

echo "=== Done ==="
echo "Checkpoint: ${CKPT}"
echo "Test metrics: ${EVAL_OUT}/metrics.json"
echo "Analysis: ${ANALYSIS_OUT}/interaction_summary.json"
