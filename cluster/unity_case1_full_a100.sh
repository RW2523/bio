#!/bin/bash
# =============================================================================
# UMass Unity Slurm — COMPLETE Case 1 pipeline (entire project path for Case 1)
#
# Runs end-to-end on one GPU node (A100):
#   1) Dataset audit
#   2) Manifest
#   3) Full preprocess (all subjects from configs/data.yaml after patch)
#   4) Case 1 train — frozen random spiking ResNet + trainable probe (configs/case1_probe.yaml)
#   5) Test-set evaluation (metrics, confusion matrix, classification report)
#
# Does NOT run: SSL pretrain, Case 2.
#
# Unity references:
#   https://docs.unity.rc.umass.edu/documentation/tools/gpus/
#   https://docs.unity.rc.umass.edu/documentation/jobs/sbatch
#
# ------------------------------------------------------------------------------
# BEFORE FIRST RUN (login node)
#
#   1) Clone repo, create venv, install deps (CUDA PyTorch matching cluster):
#        cd /path/to/bio
#        python3 -m venv .venv && source .venv/bin/activate
#        pip install -r requirements.txt
#        # Install torch with CUDA from https://pytorch.org if needed
#
#   2) Place WISDM extract so it contains: raw/phone/accel/..., activity_key.txt
#
#   3) Submit (recommended: cd into repo so logs land next to code):
#        cd /path/to/bio
#        export WISDM_DATA_ROOT=/path/to/wisdm-dataset
#        sbatch cluster/unity_case1_full_a100.sh
#
#   If you submit from another directory, set PROJECT_ROOT explicitly:
#        export PROJECT_ROOT=/path/to/bio
#        export WISDM_DATA_ROOT=/path/to/wisdm-dataset
#        sbatch --chdir="$PROJECT_ROOT" cluster/unity_case1_full_a100.sh
#
#   Override Slurm limits if Unity rejects the request:
#        sbatch --time=1-00:00:00 --mem=256G cluster/unity_case1_full_a100.sh
#
# ------------------------------------------------------------------------------
# OUTPUTS (under ${PROJECT_ROOT}/outputs/)
#
#   outputs/audit/                  — inspect_dataset + manifest
#   outputs/artifacts/              — splits, norm_stats, label_map, preprocess_run.json
#   outputs/cache/wisdm_windows/    — per-subject window .npz
#   outputs/checkpoints/case1/      — best.pt, last.pt, train.log, curves.json
#   outputs/eval_runs/case1/        — metrics.json, classification_report.txt, confusion_matrix.png
#   slurm-case1-full-<jobid>.out/.err — this job’s stdout/stderr (submit directory)
# =============================================================================

#SBATCH --job-name=wisdm-case1-full
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --gpus=1
#SBATCH --constraint=a100
#SBATCH --qos=short
#SBATCH --output=outputs/logs/slurm-case1-full-%j.out
#SBATCH --error=outputs/logs/slurm-case1-full-%j.err

set -euo pipefail

# ------------- Required: WISDM tree -----------------------------------------
# Repo root: directory that contains train/, configs/, data_tools/, eval/
: "${PROJECT_ROOT:=${SLURM_SUBMIT_DIR:-}}"
DEFAULT_WISDM_DATA_ROOT="$(dirname "${PROJECT_ROOT}")/wisdm-dataset"
: "${WISDM_DATA_ROOT:=${DEFAULT_WISDM_DATA_ROOT}}"

if [[ "${WISDM_DATA_ROOT}" == "/absolute/path/to/wisdm-dataset" ]]; then
  echo "ERROR: WISDM_DATA_ROOT is still the placeholder path /absolute/path/to/wisdm-dataset." >&2
  echo "       Use ${DEFAULT_WISDM_DATA_ROOT} for this repo layout, or pass your real dataset path." >&2
  exit 1
fi

if [[ ! -f "${PROJECT_ROOT}/train/train_linear_probe_case1.py" ]]; then
  echo "ERROR: PROJECT_ROOT must be the repository root. Current: ${PROJECT_ROOT}" >&2
  echo "       cd into the repo before sbatch, or: export PROJECT_ROOT=/absolute/path/to/repo" >&2
  exit 1
fi

if [[ ! -d "${WISDM_DATA_ROOT}" ]]; then
  echo "ERROR: WISDM_DATA_ROOT=${WISDM_DATA_ROOT} does not exist." >&2
  echo "       Expected a directory containing raw/ and activity_key.txt." >&2
  echo "       For this repo layout, the default is ${DEFAULT_WISDM_DATA_ROOT}." >&2
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
mkdir -p outputs/audit outputs/artifacts outputs/logs outputs/eval_runs/case1 outputs/checkpoints/case1

# shellcheck source=/dev/null
source "${VENV_ROOT}/bin/activate"

# Optional: match system CUDA/cuDNN to your torch build (module spider cuda)
# module load cuda/12.6
# module load cudnn/8.9.7.29-12-cuda12.6

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"

echo "=== Job ==="
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

# --- Verify merged configs (no debug subject cap; case1 epochs reasonable) --
python3 <<'VERIFY' || exit 1
import os
import sys
from pathlib import Path

pr = Path(os.environ["PROJECT_ROOT"])
sys.path.insert(0, str(pr))
os.chdir(pr)
from utils.yaml_config import load_merged_config

cfg = load_merged_config("preprocess")
if cfg.get("max_subjects") is not None:
    raise SystemExit(f"Refusing: merged preprocess has max_subjects={cfg.get('max_subjects')!r}")
if cfg.get("debug_subjects"):
    raise SystemExit(f"Refusing: merged preprocess has debug_subjects={cfg.get('debug_subjects')!r}")
print("VERIFY OK: preprocess uses all discovered subjects.")

c1 = load_merged_config("case1_probe")
print(
    "Case1 config:",
    "epochs=", c1.get("epochs"),
    "batch_size=", c1.get("batch_size"),
    "probe_optimizer=", c1.get("probe_optimizer", "sgd"),
    "lr=", c1.get("lr"),
    "compute_device=", c1.get("compute_device"),
)
if int(c1.get("epochs", 0)) < 2:
    print("WARNING: case1_probe epochs < 2 is only for debugging.", file=sys.stderr)
VERIFY

echo "=== GPU ==="
nvidia-smi -L || true

OUT_AUDIT="${PROJECT_ROOT}/outputs/audit"
ART_DIR="${PROJECT_ROOT}/outputs/artifacts"

echo "=== 1/5 Dataset audit ==="
python data_tools/inspect_dataset.py --data_root "${WISDM_DATA_ROOT}" --out_dir "${OUT_AUDIT}"

echo "=== 2/5 Manifest ==="
python data_tools/build_manifest.py --data_root "${WISDM_DATA_ROOT}" --out_dir "${OUT_AUDIT}"

echo "=== 3/5 Preprocess (full cohort; --config preprocess) ==="
python data_tools/preprocess_wisdm.py --config preprocess

echo "=== 4/5 Train Case 1 (--config case1_probe) ==="
python train/train_linear_probe_case1.py --config case1_probe

echo "=== 5/5 Evaluate held-out test subjects ==="
python eval/evaluate.py \
  --checkpoint "${PROJECT_ROOT}/outputs/checkpoints/case1/best.pt" \
  --artifacts_dir "${ART_DIR}" \
  --config case1_probe \
  --output_dir "${PROJECT_ROOT}/outputs/eval_runs/case1"

echo ""
echo "=== Case 1 pipeline finished OK ==="
echo "  Checkpoints: ${PROJECT_ROOT}/outputs/checkpoints/case1/"
echo "  Metrics:     ${PROJECT_ROOT}/outputs/eval_runs/case1/metrics.json"
echo "  Report:      ${PROJECT_ROOT}/outputs/eval_runs/case1/classification_report.txt"
echo "  Figure:      ${PROJECT_ROOT}/outputs/eval_runs/case1/confusion_matrix.png"
