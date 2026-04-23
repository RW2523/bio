#!/bin/bash
# =============================================================================
# UMass Unity Slurm — modular CNN baseline pipeline
#
# This is a switchboard for the continuous CNN ResNet baseline. Control which
# pieces run with environment flags at sbatch time.
#
# Examples:
#   # CNN Case 1 only, including preprocess and eval
#   sbatch --export=ALL,RUN_CASE1=1,RUN_EVAL_CASE1=1 cluster/unity_cnn_pipeline_a100.sh
#
#   # CNN SSL only
#   sbatch --export=ALL,RUN_SSL=1,RUN_CASE1=0,RUN_EVAL_CASE1=0 cluster/unity_cnn_pipeline_a100.sh
#
#   # CNN Case 2 using an existing outputs/checkpoints/cnn_ssl/backbone_best.pt
#   sbatch --export=ALL,RUN_PREPROCESS=0,RUN_CASE1=0,RUN_CASE2=1,RUN_EVAL_CASE2=1 cluster/unity_cnn_pipeline_a100.sh
#
#   # CNN SSL followed by Case 2 and eval
#   sbatch --export=ALL,RUN_SSL=1,RUN_CASE1=0,RUN_CASE2=1,RUN_EVAL_CASE2=1 cluster/unity_cnn_pipeline_a100.sh
#
# Flags default to a Case 1 sanity baseline:
#   RUN_AUDIT=1 RUN_MANIFEST=1 RUN_PREPROCESS=1 RUN_SSL=0
#   RUN_CASE1=1 RUN_CASE2=0 RUN_EVAL_CASE1=1 RUN_EVAL_CASE2=0
#
# Required/optional env:
#   PROJECT_ROOT=/path/to/bio
#   WISDM_DATA_ROOT=/path/to/wisdm-dataset
#   VENV_ROOT=/path/to/bio/.venv
# =============================================================================

#SBATCH --job-name=wisdm-cnn-pipe
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=03:00:00
#SBATCH --gpus=1
#SBATCH --constraint=a100
#SBATCH --qos=short
#SBATCH --output=outputs/logs/slurm-cnn-pipeline-%j.out
#SBATCH --error=outputs/logs/slurm-cnn-pipeline-%j.err

set -euo pipefail

: "${PROJECT_ROOT:=${SLURM_SUBMIT_DIR:-}}"
DEFAULT_WISDM_DATA_ROOT="$(dirname "${PROJECT_ROOT}")/wisdm-dataset"
: "${WISDM_DATA_ROOT:=${DEFAULT_WISDM_DATA_ROOT}}"
: "${VENV_ROOT:=${PROJECT_ROOT}/.venv}"

: "${RUN_AUDIT:=1}"
: "${RUN_MANIFEST:=1}"
: "${RUN_PREPROCESS:=1}"
: "${RUN_SSL:=0}"
: "${RUN_CASE1:=1}"
: "${RUN_CASE2:=0}"
: "${RUN_EVAL_CASE1:=1}"
: "${RUN_EVAL_CASE2:=0}"

export PROJECT_ROOT WISDM_DATA_ROOT VENV_ROOT

if [[ ! -f "${PROJECT_ROOT}/train/pretrain_augpred_cnn.py" ]]; then
  echo "ERROR: PROJECT_ROOT must be the repository root. Current: ${PROJECT_ROOT}" >&2
  exit 1
fi

if [[ ! -d "${WISDM_DATA_ROOT}/raw" || ! -f "${WISDM_DATA_ROOT}/activity_key.txt" ]]; then
  echo "ERROR: WISDM_DATA_ROOT=${WISDM_DATA_ROOT} must contain raw/ and activity_key.txt." >&2
  exit 1
fi

if [[ ! -f "${VENV_ROOT}/bin/activate" ]]; then
  echo "ERROR: Missing venv at ${VENV_ROOT}" >&2
  exit 1
fi

cd "${PROJECT_ROOT}"
mkdir -p outputs/audit outputs/artifacts outputs/logs \
  outputs/checkpoints/cnn_ssl outputs/checkpoints/cnn_case1 outputs/checkpoints/cnn_case2 \
  outputs/eval_runs/cnn_case1 outputs/eval_runs/cnn_case2

# shellcheck source=/dev/null
source "${VENV_ROOT}/bin/activate"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"

echo "=== CNN modular pipeline ==="
echo "PROJECT_ROOT=${PROJECT_ROOT}"
echo "WISDM_DATA_ROOT=${WISDM_DATA_ROOT}"
echo "RUN_AUDIT=${RUN_AUDIT} RUN_MANIFEST=${RUN_MANIFEST} RUN_PREPROCESS=${RUN_PREPROCESS}"
echo "RUN_SSL=${RUN_SSL} RUN_CASE1=${RUN_CASE1} RUN_CASE2=${RUN_CASE2}"
echo "RUN_EVAL_CASE1=${RUN_EVAL_CASE1} RUN_EVAL_CASE2=${RUN_EVAL_CASE2}"
python - <<'VERS'
import sys
import torch
print("python", sys.version.split()[0], "| torch", torch.__version__, "| cuda?", torch.cuda.is_available())
VERS

DATA_YAML="${PROJECT_ROOT}/configs/data.yaml"
DATA_BAK="${SLURM_TMPDIR:-/tmp}/data.yaml.bak.${SLURM_JOB_ID:-$$}"
cp -a "${DATA_YAML}" "${DATA_BAK}"
restore_data_yaml() { cp -a "${DATA_BAK}" "${DATA_YAML}"; }
trap restore_data_yaml EXIT

python3 <<'PATCH'
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
p.write_text(yaml.dump(cfg, sort_keys=False), encoding="utf-8")
print(f"Patched data.yaml: data_root={dr} max_subjects=null debug_subjects=[]")
PATCH

python3 <<'VERIFY'
from utils.yaml_config import load_merged_config
for name in ["ssl_cnn", "case1_probe_cnn", "case2_probe_cnn"]:
    cfg = load_merged_config(name)
    print(
        name,
        "epochs=", cfg.get("epochs"),
        "batch_size=", cfg.get("batch_size"),
        "checkpoint_subdir=", cfg.get("checkpoint_subdir"),
        "device=", cfg.get("compute_device"),
    )
VERIFY

ART_DIR="${PROJECT_ROOT}/outputs/artifacts"

if [[ "${RUN_AUDIT}" == "1" ]]; then
  echo "=== Dataset audit ==="
  python data_tools/inspect_dataset.py --data_root "${WISDM_DATA_ROOT}" --out_dir "${PROJECT_ROOT}/outputs/audit"
fi

if [[ "${RUN_MANIFEST}" == "1" ]]; then
  echo "=== Manifest ==="
  python data_tools/build_manifest.py --data_root "${WISDM_DATA_ROOT}" --out_dir "${PROJECT_ROOT}/outputs/audit"
fi

if [[ "${RUN_PREPROCESS}" == "1" ]]; then
  echo "=== Preprocess ==="
  python data_tools/preprocess_wisdm.py --config preprocess
fi

if [[ "${RUN_SSL}" == "1" ]]; then
  echo "=== CNN AugPred SSL ==="
  python train/pretrain_augpred_cnn.py --config ssl_cnn
fi

if [[ "${RUN_CASE1}" == "1" ]]; then
  echo "=== CNN Case 1: random frozen backbone + probe ==="
  python train/train_linear_probe_case1_cnn.py --config case1_probe_cnn
fi

if [[ "${RUN_CASE2}" == "1" ]]; then
  if [[ ! -f "${PROJECT_ROOT}/outputs/checkpoints/cnn_ssl/backbone_best.pt" ]]; then
    echo "ERROR: missing CNN SSL backbone at outputs/checkpoints/cnn_ssl/backbone_best.pt" >&2
    echo "Run with RUN_SSL=1 first, or point configs/case2_probe_cnn.yaml at an existing checkpoint." >&2
    exit 1
  fi
  echo "=== CNN Case 2: AugPred-pretrained frozen backbone + probe ==="
  python train/train_linear_probe_case2_cnn.py --config case2_probe_cnn
fi

if [[ "${RUN_EVAL_CASE1}" == "1" ]]; then
  echo "=== Evaluate CNN Case 1 ==="
  python eval/evaluate_cnn.py \
    --checkpoint "${PROJECT_ROOT}/outputs/checkpoints/cnn_case1/best.pt" \
    --artifacts_dir "${ART_DIR}" \
    --config case1_probe_cnn \
    --output_dir "${PROJECT_ROOT}/outputs/eval_runs/cnn_case1"
fi

if [[ "${RUN_EVAL_CASE2}" == "1" ]]; then
  echo "=== Evaluate CNN Case 2 ==="
  python eval/evaluate_cnn.py \
    --checkpoint "${PROJECT_ROOT}/outputs/checkpoints/cnn_case2/best.pt" \
    --artifacts_dir "${ART_DIR}" \
    --config case2_probe_cnn \
    --output_dir "${PROJECT_ROOT}/outputs/eval_runs/cnn_case2"
fi

echo "=== CNN modular pipeline finished ==="
