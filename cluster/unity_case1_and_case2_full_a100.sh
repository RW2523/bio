#!/bin/bash
# =============================================================================
# UMass Unity Slurm — FULL Case 1 + Case 2 (single job, one GPU)
#
# Order (matches professor design):
#   1) Dataset audit
#   2) Manifest
#   3) Full preprocess → outputs/artifacts + window cache
#   4) AugPred SSL (configs/ssl_20epochs.yaml) → outputs/checkpoints/ssl_ep20/backbone_best.pt
#   5) Case 1 — random frozen spiking ResNet + linear probe (configs/cluster_case1_100ep.yaml, 100 epochs)
#   6) Evaluate Case 1 on held-out test
#   7) Case 2 — SSL-pretrained frozen backbone + same probe recipe (configs/cluster_case2_ssl20_100ep.yaml)
#   8) Evaluate Case 2 on held-out test
#
# SSL uses 20 epochs (not 5) for stronger representations; probes use 100 epochs each.
#
# BEFORE SUBMIT (login node):
#   cd /path/to/bio
#   mkdir -p outputs/logs
#   python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
#   export WISDM_DATA_ROOT=/path/to/wisdm-dataset   # if not sibling ../wisdm-dataset
#   sbatch cluster/unity_case1_and_case2_full_a100.sh
#
# Long walltime override if Unity rejects default:
#   sbatch --time=2-00:00:00 --mem=256G cluster/unity_case1_and_case2_full_a100.sh
#
# Docs: https://docs.unity.rc.umass.edu/documentation/jobs/sbatch
# =============================================================================

#SBATCH --job-name=wisdm-case1-case2-full
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --gpus=1
#SBATCH --constraint=a100
#SBATCH --qos=short
#SBATCH --output=outputs/logs/slurm-case1-case2-full-%j.out
#SBATCH --error=outputs/logs/slurm-case1-case2-full-%j.err

set -euo pipefail

: "${PROJECT_ROOT:=${SLURM_SUBMIT_DIR:-}}"
DEFAULT_WISDM_DATA_ROOT="$(dirname "${PROJECT_ROOT}")/wisdm-dataset"
: "${WISDM_DATA_ROOT:=${DEFAULT_WISDM_DATA_ROOT}}"

if [[ ! -f "${PROJECT_ROOT}/train/train_linear_probe_case1.py" ]] || [[ ! -f "${PROJECT_ROOT}/train/train_linear_probe_case2.py" ]]; then
  echo "ERROR: PROJECT_ROOT must be the bio repository root. Got: ${PROJECT_ROOT}" >&2
  echo "  cd into the repo before sbatch, or: export PROJECT_ROOT=/absolute/path/to/bio" >&2
  exit 1
fi

if [[ ! -d "${WISDM_DATA_ROOT}/raw" || ! -f "${WISDM_DATA_ROOT}/activity_key.txt" ]]; then
  echo "ERROR: WISDM_DATA_ROOT=${WISDM_DATA_ROOT} must contain raw/ and activity_key.txt" >&2
  exit 1
fi

: "${VENV_ROOT:=${PROJECT_ROOT}/.venv}"
export PROJECT_ROOT WISDM_DATA_ROOT VENV_ROOT

if [[ ! -f "${VENV_ROOT}/bin/activate" ]]; then
  echo "ERROR: Missing venv at ${VENV_ROOT}" >&2
  exit 1
fi

cd "${PROJECT_ROOT}"
mkdir -p outputs/audit outputs/artifacts outputs/logs \
  outputs/cache/wisdm_windows \
  outputs/checkpoints/ssl_ep20 outputs/checkpoints/cluster_case1 outputs/checkpoints/cluster_case2 \
  outputs/eval_runs/cluster_case1 outputs/eval_runs/cluster_case2

# shellcheck source=/dev/null
source "${VENV_ROOT}/bin/activate"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"

echo "=== Case1+Case2 full job ==="
echo "PROJECT_ROOT=${PROJECT_ROOT}"
echo "WISDM_DATA_ROOT=${WISDM_DATA_ROOT}"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-local} HOST=$(hostname)"
python - <<'VERS' || exit 1
import sys
import torch
print("python", sys.version.split()[0], "| torch", torch.__version__, "| cuda?", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("CUDA required: configs use compute_device cuda")
VERS

nvidia-smi -L || true

# --- Patch configs/data.yaml (restore on exit) ----------------------------
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
print("Patched data.yaml data_root ->", dr)
PATCH

# --- Config sanity ---------------------------------------------------------
python3 <<'VERIFY' || exit 1
import os, sys
from pathlib import Path
pr = Path(os.environ["PROJECT_ROOT"])
sys.path.insert(0, str(pr))
os.chdir(pr)
from utils.yaml_config import load_merged_config
from train.common import resolve_path

for name in ("preprocess", "ssl_20epochs", "cluster_case1_100ep", "cluster_case2_ssl20_100ep"):
    load_merged_config(name)
    print("OK merged:", name)

ssl = load_merged_config("ssl_20epochs")
assert int(ssl.get("epochs", 0)) >= 20, ssl

c1 = load_merged_config("cluster_case1_100ep")
assert int(c1.get("epochs", 0)) >= 50, c1

c2 = load_merged_config("cluster_case2_ssl20_100ep")
bb = resolve_path(str(c2["pretrained_backbone_path"]))
print("Case2 will load SSL backbone from:", bb)
assert "ssl_ep20" in str(bb) or "backbone_best" in str(bb), bb
VERIFY

OUT_AUDIT="${PROJECT_ROOT}/outputs/audit"
ART_DIR="${PROJECT_ROOT}/outputs/artifacts"
SSL_BB="${PROJECT_ROOT}/outputs/checkpoints/ssl_ep20/backbone_best.pt"
C1_CKPT="${PROJECT_ROOT}/outputs/checkpoints/cluster_case1/best.pt"
C2_CKPT="${PROJECT_ROOT}/outputs/checkpoints/cluster_case2/best.pt"

echo "=== 1/8 Dataset audit ==="
python data_tools/inspect_dataset.py --data_root "${WISDM_DATA_ROOT}" --out_dir "${OUT_AUDIT}"

echo "=== 2/8 Manifest ==="
python data_tools/build_manifest.py --data_root "${WISDM_DATA_ROOT}" --out_dir "${OUT_AUDIT}"

echo "=== 3/8 Preprocess (full cohort) ==="
python data_tools/preprocess_wisdm.py --config preprocess

echo "=== 4/8 SSL pretrain (20 epochs; ssl_20epochs) ==="
python train/pretrain_augpred.py --config ssl_20epochs

if [[ ! -f "${SSL_BB}" ]]; then
  echo "ERROR: Missing ${SSL_BB} after SSL. Check outputs/checkpoints/ssl_ep20/" >&2
  exit 1
fi
echo "OK SSL backbone: ${SSL_BB}"

echo "=== 5/8 Case 1 — random frozen backbone + linear probe (100 epochs) ==="
python train/train_linear_probe_case1.py --config cluster_case1_100ep

if [[ ! -f "${C1_CKPT}" ]]; then
  echo "ERROR: Missing ${C1_CKPT}" >&2
  exit 1
fi

echo "=== 6/8 Evaluate Case 1 (test subjects) ==="
python eval/evaluate.py \
  --checkpoint "${C1_CKPT}" \
  --artifacts_dir "${ART_DIR}" \
  --config model \
  --output_dir "${PROJECT_ROOT}/outputs/eval_runs/cluster_case1"

echo "=== 7/8 Case 2 — SSL frozen backbone + linear probe (100 epochs) ==="
python train/train_linear_probe_case2.py --config cluster_case2_ssl20_100ep

if [[ ! -f "${C2_CKPT}" ]]; then
  echo "ERROR: Missing ${C2_CKPT}" >&2
  exit 1
fi

echo "=== 8/8 Evaluate Case 2 (test subjects) ==="
python eval/evaluate.py \
  --checkpoint "${C2_CKPT}" \
  --artifacts_dir "${ART_DIR}" \
  --config model \
  --output_dir "${PROJECT_ROOT}/outputs/eval_runs/cluster_case2"

echo ""
echo "=== ALL STEPS FINISHED OK ==="
echo "Case 1:  ${C1_CKPT}"
echo "  metrics: ${PROJECT_ROOT}/outputs/eval_runs/cluster_case1/metrics.json"
echo "Case 2:  ${C2_CKPT}"
echo "  metrics: ${PROJECT_ROOT}/outputs/eval_runs/cluster_case2/metrics.json"
echo "SSL:     ${SSL_BB}"
