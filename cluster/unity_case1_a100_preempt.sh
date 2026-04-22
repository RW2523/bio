#!/bin/bash
# =============================================================================
# UMass Unity (Slurm) — Case 1 QUICK / preempt-friendly run (short walltime + qos short)
#
# For the ENTIRE cohort + multi-epoch Case 1 (production), use instead:
#   cluster/unity_case1_full_a100.sh
#
# UMass Unity (Slurm) — WISDM pipeline for Case 1 only (no SSL / Case 2)
#
# Docs referenced:
#   https://docs.unity.rc.umass.edu/documentation/tools/gpus/
#   https://docs.unity.rc.umass.edu/documentation/jobs/sbatch
#   Unity quick start: preempt partition + --qos=short for short jobs
#
# Submit from the repo root (directory that contains train/, configs/, …), e.g.:
#   cd /path/to/project
#   export WISDM_DATA_ROOT=/path/to/wisdm-dataset
#   sbatch cluster/unity_case1_a100_preempt.sh
#
# Or pass the variable at submit time:
#   sbatch --export=ALL,WISDM_DATA_ROOT=/path/to/wisdm-dataset cluster/unity_case1_a100_preempt.sh
#
# Optional: pin the working directory explicitly (uncomment and edit):
# #SBATCH --chdir=/path/to/project
#
# Notes:
#   - gpu-preempt: higher-priority jobs may preempt yours; prefer walltime <= 2h
#     for the least disruption, or use checkpointing / resubmit if killed.
#   - --qos=short: Unity documents a priority boost for jobs under ~4 hours.
#   - Request A100 via --constraint=a100 (see Unity GPU constraint table).
#   - If preprocess + training exceed -t, raise -t (still < 4h to keep --qos=short)
#     or drop --qos=short and/or use partition gpu (non-preempt, longer waits possible).
# =============================================================================

#SBATCH --job-name=wisdm-case1
#SBATCH --partition=gpu-preempt
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:30:00
#SBATCH --gpus=1
#SBATCH --constraint=a100
#SBATCH --output=slurm-case1-%j.out
#SBATCH --error=slurm-case1-%j.err

set -euo pipefail

# ------------- EDIT THESE (or export before sbatch) --------------------------
# Directory that contains train/, configs/, data_tools/, ... (this repo layout)
: "${PROJECT_ROOT:=${SLURM_SUBMIT_DIR}}"
# Extracted WISDM tree: must contain raw/, activity_key.txt
if [[ -z "${WISDM_DATA_ROOT:-}" ]]; then
  echo "ERROR: export WISDM_DATA_ROOT=/path/to/wisdm-dataset before sbatch (see script header)." >&2
  exit 1
fi
# Python venv with requirements.txt installed (CUDA PyTorch build recommended)
: "${VENV_ROOT:=${PROJECT_ROOT}/.venv}"
# ---------------------------------------------------------------------------

cd "${PROJECT_ROOT}"
if [[ ! -f "train/train_linear_probe_case1.py" ]]; then
  echo "ERROR: PROJECT_ROOT=${PROJECT_ROOT} does not look like the project repo root." >&2
  exit 1
fi

# shellcheck source=/dev/null
source "${VENV_ROOT}/bin/activate"

# Optional: load CUDA stack matching your torch build (versions from: module spider cuda)
# module load conda/latest   # if you use modules instead of a venv
# module load cuda/12.6
# module load cudnn/8.9.7.29-12-cuda12.6

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

# Temporarily point configs/data.yaml at WISDM (restored on exit)
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

OUT_AUDIT="${PROJECT_ROOT}/outputs/audit"
mkdir -p "${OUT_AUDIT}"

echo "=== 1) Dataset audit ==="
python data_tools/inspect_dataset.py --data_root "${WISDM_DATA_ROOT}" --out_dir "${OUT_AUDIT}"

echo "=== 2) Manifest ==="
python data_tools/build_manifest.py --data_root "${WISDM_DATA_ROOT}" --out_dir "${OUT_AUDIT}"

echo "=== 3) Preprocess (full dataset; configs/preprocess + data) ==="
python data_tools/preprocess_wisdm.py --config preprocess

echo "=== 4) Case 1 linear probe (frozen random backbone) ==="
python train/train_linear_probe_case1.py --config case1_probe

echo "=== 5) Evaluate Case 1 checkpoint ==="
python eval/evaluate.py \
  --checkpoint "${PROJECT_ROOT}/outputs/checkpoints/case1/best.pt" \
  --output_dir "${PROJECT_ROOT}/outputs/eval_runs/case1"

echo "=== Done. Artifacts under ${PROJECT_ROOT}/outputs/ ==="
