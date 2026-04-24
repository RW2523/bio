#!/bin/bash
# =============================================================================
# UMass Unity Slurm — FULL WISDM pipeline (clean run, single GPU)
#
# Order:
#   1) Dataset audit
#   2) Manifest
#   3) Full preprocess → artifacts + window cache under OUTPUT_ROOT
#   4) AugPred SSL (20 epochs) → OUTPUT_ROOT/checkpoints/ssl_ep20/backbone_best.pt
#   5) Case 1 — random frozen SNN + linear probe (100 epochs)
#   6) Evaluate Case 1 (held-out test)
#   7) Case 2 — SSL-pretrained frozen backbone + linear probe (100 epochs)
#   8) Evaluate Case 2 (held-out test)
#
# All outputs go under a NEW folder only (never reuses outputs/, outputs 2/, or old runs):
#   outputs_unity_full_run_${SLURM_JOB_ID}
# If SLURM_JOB_ID is unset (e.g. local): outputs_unity_full_run_local
#
# Resource hints:
#   - If 12 hours is not enough: sbatch --time=2-00:00:00 cluster/unity_case1_and_case2_full_a100.sh
#   - If memory is tight:        sbatch --mem=256G cluster/unity_case1_and_case2_full_a100.sh
#   - If no A100:               remove #SBATCH --constraint=a100 (or pass overrides)
#
# BEFORE SUBMIT:
#   cd /path/to/bio
#   python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
#   export WISDM_DATA_ROOT=/path/to/wisdm-dataset
#   sbatch cluster/unity_case1_and_case2_full_a100.sh
#
# Docs: https://docs.unity.rc.umass.edu/documentation/jobs/sbatch
# =============================================================================

#SBATCH --job-name=wisdm-unity-full
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --gpus=1
#SBATCH --constraint=a100
#SBATCH --qos=short
#SBATCH --output=slurm-unity-full-%j.out
#SBATCH --error=slurm-unity-full-%j.err

set -euo pipefail

# ---------------------------------------------------------------------------
# PROJECT_ROOT: SLURM_SUBMIT_DIR is the cwd when sbatch was run (Unity: use "cd bio" first).
# If that fails, use this script's parent dir ONLY when it actually contains train/ (Slurm may
# execute a spool copy of the script — then script path is NOT under the repo).
# ---------------------------------------------------------------------------
_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_REPO_CANDIDATE="$(cd "${_SCRIPT_DIR}/.." && pwd)"
if [[ -f "${_REPO_CANDIDATE}/train/train_linear_probe_case1.py" ]]; then
  _REPO_FROM_SCRIPT="${_REPO_CANDIDATE}"
else
  _REPO_FROM_SCRIPT=""
fi

: "${PROJECT_ROOT:=${SLURM_SUBMIT_DIR:-}}"
PROJECT_ROOT="${PROJECT_ROOT%/}"
if [[ ! -f "${PROJECT_ROOT}/train/train_linear_probe_case1.py" ]]; then
  if [[ -n "${_REPO_FROM_SCRIPT}" ]]; then
    PROJECT_ROOT="${_REPO_FROM_SCRIPT}"
  fi
fi
PROJECT_ROOT="${PROJECT_ROOT%/}"

if [[ ! -f "${PROJECT_ROOT}/train/train_linear_probe_case1.py" ]] || [[ ! -f "${PROJECT_ROOT}/train/train_linear_probe_case2.py" ]]; then
  echo "ERROR: Cannot find bio repo root (train scripts missing)." >&2
  echo "  PROJECT_ROOT=${PROJECT_ROOT:-<empty>}" >&2
  echo "  SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR:-<empty>}" >&2
  echo "  Script: ${BASH_SOURCE[0]} (candidate from script: ${_REPO_FROM_SCRIPT:-<none>})" >&2
  echo "  Fix: cd /path/to/bio && sbatch cluster/unity_case1_and_case2_full_a100.sh" >&2
  exit 1
fi

DEFAULT_WISDM_DATA_ROOT="$(dirname "${PROJECT_ROOT}")/wisdm-dataset"
: "${WISDM_DATA_ROOT:=${DEFAULT_WISDM_DATA_ROOT}}"

if [[ ! -d "${WISDM_DATA_ROOT}/raw" ]]; then
  echo "ERROR: WISDM_DATA_ROOT=${WISDM_DATA_ROOT} must contain directory raw/" >&2
  exit 1
fi
if [[ ! -f "${WISDM_DATA_ROOT}/activity_key.txt" ]]; then
  echo "ERROR: WISDM_DATA_ROOT=${WISDM_DATA_ROOT} must contain file activity_key.txt" >&2
  exit 1
fi

: "${VENV_ROOT:=${PROJECT_ROOT}/.venv}"
if [[ ! -f "${VENV_ROOT}/bin/activate" ]]; then
  echo "ERROR: Missing virtualenv at ${VENV_ROOT} (expected .venv from repo root)." >&2
  exit 1
fi

# ---------------------------------------------------------------------------
# Fresh output root (never outputs/ or outputs 2/)
# ---------------------------------------------------------------------------
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  OUTPUT_ROOT="outputs_unity_full_run_${SLURM_JOB_ID}"
else
  OUTPUT_ROOT="outputs_unity_full_run_local"
fi

for forbidden in "outputs" "outputs 2"; do
  if [[ "${OUTPUT_ROOT}" == "${forbidden}" ]]; then
    echo "ERROR: OUTPUT_ROOT must not be legacy path '${forbidden}'." >&2
    exit 1
  fi
done

export PROJECT_ROOT WISDM_DATA_ROOT VENV_ROOT OUTPUT_ROOT

ABS_OUT="${PROJECT_ROOT}/${OUTPUT_ROOT}"
STEP_LOG="${ABS_OUT}/logs/job_steps.log"
RUNTIME_ENV="${ABS_OUT}/logs/runtime_env.txt"
START_TS=$(date +%s)

mkdir -p "${ABS_OUT}/audit" "${ABS_OUT}/artifacts" "${ABS_OUT}/logs" \
  "${ABS_OUT}/cache/wisdm_windows" \
  "${ABS_OUT}/checkpoints/ssl_ep20" "${ABS_OUT}/checkpoints/cluster_case1" "${ABS_OUT}/checkpoints/cluster_case2" \
  "${ABS_OUT}/eval_runs/cluster_case1" "${ABS_OUT}/eval_runs/cluster_case2"

: >"${STEP_LOG}"

# ---------------------------------------------------------------------------
# run_step: log boundaries; run command with stdout+stderr copied to STEP_LOG
# ---------------------------------------------------------------------------
run_step() {
  local name="$1"
  shift
  echo "=== START: ${name} ===" | tee -a "${STEP_LOG}"
  date -Is | tee -a "${STEP_LOG}"
  set +e
  "$@" 2>&1 | tee -a "${STEP_LOG}"
  local st=${PIPESTATUS[0]}
  set -e
  date -Is | tee -a "${STEP_LOG}"
  echo "=== END: ${name} (exit=${st}) ===" | tee -a "${STEP_LOG}"
  if [[ "${st}" -ne 0 ]]; then
    echo "ERROR: Step '${name}' failed with exit ${st}. See ${STEP_LOG}" >&2
    exit "${st}"
  fi
}

# ---------------------------------------------------------------------------
# Config backup + restore on EXIT (even on failure)
# ---------------------------------------------------------------------------
CFG_BAK_ROOT="${SLURM_TMPDIR:-/tmp}/unity_cfg_bak_${SLURM_JOB_ID:-$$}_${RANDOM}"
mkdir -p "${CFG_BAK_ROOT}"

CONFIG_FILES_TO_PATCH=(
  "data.yaml"
  "preprocess.yaml"
  "case2_after_ssl20.yaml"
)

backup_configs() {
  local f
  for f in "${CONFIG_FILES_TO_PATCH[@]}"; do
    local src="${PROJECT_ROOT}/configs/${f}"
    if [[ ! -f "${src}" ]]; then
      echo "ERROR: Missing required config ${src}" >&2
      exit 1
    fi
    cp -a "${src}" "${CFG_BAK_ROOT}/${f}.orig"
  done
  echo "Backed up configs to ${CFG_BAK_ROOT}" | tee -a "${STEP_LOG}"
}

restore_configs() {
  local f
  for f in "${CONFIG_FILES_TO_PATCH[@]}"; do
    local bak="${CFG_BAK_ROOT}/${f}.orig"
    local dst="${PROJECT_ROOT}/configs/${f}"
    if [[ -f "${bak}" ]]; then
      cp -a "${bak}" "${dst}" || true
    fi
  done
  echo "Restored configs from ${CFG_BAK_ROOT}" >>"${STEP_LOG}" 2>&1 || true
}

trap restore_configs EXIT
backup_configs

# ---------------------------------------------------------------------------
# Runtime environment log
# ---------------------------------------------------------------------------
{
  echo "hostname=$(hostname)"
  echo "date=$(date -Is)"
  echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
  echo "PROJECT_ROOT=${PROJECT_ROOT}"
  echo "WISDM_DATA_ROOT=${WISDM_DATA_ROOT}"
  echo "OUTPUT_ROOT=${OUTPUT_ROOT}"
  echo "ABS_OUT=${ABS_OUT}"
  echo "CFG_BAK_ROOT=${CFG_BAK_ROOT}"
  echo "--- Python ---"
  "${VENV_ROOT}/bin/python" -V 2>&1 || true
  echo "--- Torch / CUDA ---"
  "${VENV_ROOT}/bin/python" - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
print("cuda_device_count", torch.cuda.device_count())
PY
  echo "--- nvidia-smi ---"
  nvidia-smi 2>&1 || echo "(nvidia-smi not available)"
  echo "--- git ---"
  (cd "${PROJECT_ROOT}" && git rev-parse HEAD 2>/dev/null) || echo "(not a git repo or no commit)"
} >"${RUNTIME_ENV}" 2>&1

echo "=== Unity full WISDM pipeline ===" | tee -a "${STEP_LOG}"
echo "RUNTIME_ENV=${RUNTIME_ENV}" | tee -a "${STEP_LOG}"
echo "STEP_LOG=${STEP_LOG}" | tee -a "${STEP_LOG}"

# Copy Slurm stream files into OUTPUT_ROOT/logs (best effort; re-copy at end for full tail).
copy_slurm_streams_to_output() {
  if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    return 0
  fi
  for sfx in out err; do
    src="${PROJECT_ROOT}/slurm-unity-full-${SLURM_JOB_ID}.${sfx}"
    if [[ -f "${src}" ]]; then
      cp -a "${src}" "${ABS_OUT}/logs/slurm_stream.${sfx}" || true
    fi
  done
}
copy_slurm_streams_to_output

cd "${PROJECT_ROOT}"
# shellcheck source=/dev/null
source "${VENV_ROOT}/bin/activate"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"

# ---------------------------------------------------------------------------
# Patch configs (Python; stdin heredoc — cannot use run_step wrapper)
# ---------------------------------------------------------------------------
echo "=== START: patch_configs ===" | tee -a "${STEP_LOG}"
date -Is | tee -a "${STEP_LOG}"
"${VENV_ROOT}/bin/python" - <<'PATCH' 2>&1 | tee -a "${STEP_LOG}"
import os
from pathlib import Path
import yaml

pr = Path(os.environ["PROJECT_ROOT"])
dr = Path(os.environ["WISDM_DATA_ROOT"]).resolve()
out_root = os.environ["OUTPUT_ROOT"].strip().rstrip("/")

p = pr / "configs" / "data.yaml"
cfg = yaml.safe_load(p.read_text(encoding="utf-8"))
cfg["data_root"] = str(dr)
cfg["max_subjects"] = None
cfg["debug_subjects"] = []
p.write_text(yaml.dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
print("Patched data.yaml data_root ->", dr)

pp = pr / "configs" / "preprocess.yaml"
pcfg = yaml.safe_load(pp.read_text(encoding="utf-8"))
pcfg["output_dir"] = out_root
pp.write_text(yaml.dump(pcfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
print("Patched preprocess.yaml output_dir ->", out_root)

c2 = pr / "configs" / "case2_after_ssl20.yaml"
d = yaml.safe_load(c2.read_text(encoding="utf-8")) or {}
d["pretrained_backbone_path"] = f"{out_root}/checkpoints/ssl_ep20/backbone_best.pt"
c2.write_text(yaml.dump(d, sort_keys=False, allow_unicode=True), encoding="utf-8")
print("Patched case2_after_ssl20.yaml pretrained_backbone_path ->", d["pretrained_backbone_path"])
PATCH
PATCH_ST=${PIPESTATUS[0]}
date -Is | tee -a "${STEP_LOG}"
echo "=== END: patch_configs (exit=${PATCH_ST}) ===" | tee -a "${STEP_LOG}"
if [[ "${PATCH_ST}" -ne 0 ]]; then
  echo "ERROR: patch_configs failed (exit=${PATCH_ST})" >&2
  exit "${PATCH_ST}"
fi

# ---------------------------------------------------------------------------
# Required config files
# ---------------------------------------------------------------------------
REQUIRED_CFGS=(
  "configs/data.yaml"
  "configs/preprocess.yaml"
  "configs/model.yaml"
  "configs/ssl.yaml"
  "configs/ssl_20epochs.yaml"
  "configs/case1_probe.yaml"
  "configs/cluster_case1_100ep.yaml"
  "configs/case2_probe.yaml"
  "configs/case2_after_ssl20.yaml"
  "configs/cluster_case2_ssl20_100ep.yaml"
)
for rel in "${REQUIRED_CFGS[@]}"; do
  if [[ ! -f "${PROJECT_ROOT}/${rel}" ]]; then
    echo "ERROR: Missing required file ${PROJECT_ROOT}/${rel}" >&2
    exit 1
  fi
done

# ---------------------------------------------------------------------------
# Validate merged configs (heredoc)
# ---------------------------------------------------------------------------
echo "=== START: validate_merged_configs ===" | tee -a "${STEP_LOG}"
date -Is | tee -a "${STEP_LOG}"
"${VENV_ROOT}/bin/python" - <<'VERIFY' 2>&1 | tee -a "${STEP_LOG}"
import os, sys
from pathlib import Path

pr = Path(os.environ["PROJECT_ROOT"])
sys.path.insert(0, str(pr))
os.chdir(pr)
from utils.yaml_config import load_merged_config
from train.common import resolve_path

out_root = os.environ["OUTPUT_ROOT"].strip().rstrip("/")
expected_bb = (pr / out_root / "checkpoints" / "ssl_ep20" / "backbone_best.pt").resolve()

for name in ("preprocess", "ssl_20epochs", "cluster_case1_100ep", "cluster_case2_ssl20_100ep"):
    load_merged_config(name)
    print("OK merged:", name)

ssl = load_merged_config("ssl_20epochs")
ep_ssl = int(ssl.get("epochs", 0))
if ep_ssl < 20:
    raise SystemExit(f"ssl_20epochs must have epochs >= 20, got {ep_ssl}")

c1 = load_merged_config("cluster_case1_100ep")
ep_c1 = int(c1.get("epochs", 0))
if ep_c1 != 100:
    raise SystemExit(f"cluster_case1_100ep expected epochs==100, got {ep_c1}")

c2 = load_merged_config("cluster_case2_ssl20_100ep")
ep_c2 = int(c2.get("epochs", 0))
if ep_c2 != 100:
    raise SystemExit(f"cluster_case2_ssl20_100ep expected epochs==100, got {ep_c2}")

bb = resolve_path(str(c2["pretrained_backbone_path"]))
print("Case2 pretrained_backbone_path resolves to:", bb)
if bb.resolve() != expected_bb.resolve():
    raise SystemExit(f"Expected Case2 SSL path {expected_bb}, got {bb}")
VERIFY
VERIFY_ST=${PIPESTATUS[0]}
date -Is | tee -a "${STEP_LOG}"
echo "=== END: validate_merged_configs (exit=${VERIFY_ST}) ===" | tee -a "${STEP_LOG}"
if [[ "${VERIFY_ST}" -ne 0 ]]; then
  echo "ERROR: validate_merged_configs failed" >&2
  exit "${VERIFY_ST}"
fi

# ---------------------------------------------------------------------------
# CUDA before long training
# ---------------------------------------------------------------------------
echo "=== START: check_cuda_before_training ===" | tee -a "${STEP_LOG}"
date -Is | tee -a "${STEP_LOG}"
"${VENV_ROOT}/bin/python" - <<'VERS' 2>&1 | tee -a "${STEP_LOG}"
import torch
if not torch.cuda.is_available():
    raise SystemExit("CUDA required: training configs use compute_device cuda")
print("CUDA OK:", torch.cuda.get_device_name(0))
VERS
CUDA_ST=${PIPESTATUS[0]}
date -Is | tee -a "${STEP_LOG}"
echo "=== END: check_cuda_before_training (exit=${CUDA_ST}) ===" | tee -a "${STEP_LOG}"
if [[ "${CUDA_ST}" -ne 0 ]]; then
  echo "ERROR: CUDA check failed" >&2
  exit "${CUDA_ST}"
fi

ART_DIR="${ABS_OUT}/artifacts"
SSL_BB="${ABS_OUT}/checkpoints/ssl_ep20/backbone_best.pt"
C1_CKPT="${ABS_OUT}/checkpoints/cluster_case1/best.pt"
C2_CKPT="${ABS_OUT}/checkpoints/cluster_case2/best.pt"
C1_METRICS="${ABS_OUT}/eval_runs/cluster_case1/metrics.json"
C2_METRICS="${ABS_OUT}/eval_runs/cluster_case2/metrics.json"

# ---------------------------------------------------------------------------
# Pipeline (cwd = repo). Use venv Python explicitly (Unity-safe if PATH is odd).
# ---------------------------------------------------------------------------
PY="${VENV_ROOT}/bin/python"
run_step "01_dataset_audit" \
  "${PY}" data_tools/inspect_dataset.py --data_root "${WISDM_DATA_ROOT}" --out_dir "${OUTPUT_ROOT}/audit"

run_step "02_build_manifest" \
  "${PY}" data_tools/build_manifest.py --data_root "${WISDM_DATA_ROOT}" --out_dir "${OUTPUT_ROOT}/audit"

run_step "03_preprocess_full" \
  "${PY}" data_tools/preprocess_wisdm.py --config preprocess

run_step "04_ssl_pretrain_20ep" \
  "${PY}" train/pretrain_augpred.py --config ssl_20epochs

if [[ ! -f "${SSL_BB}" ]]; then
  echo "ERROR: Missing SSL checkpoint after step 4: ${SSL_BB}" | tee -a "${STEP_LOG}" >&2
  exit 1
fi

run_step "05_case1_linear_probe_100ep" \
  "${PY}" train/train_linear_probe_case1.py --config cluster_case1_100ep

if [[ ! -f "${C1_CKPT}" ]]; then
  echo "ERROR: Missing Case1 checkpoint: ${C1_CKPT}" | tee -a "${STEP_LOG}" >&2
  exit 1
fi

run_step "06_evaluate_case1" \
  "${PY}" eval/evaluate.py \
    --checkpoint "${C1_CKPT}" \
    --artifacts_dir "${ART_DIR}" \
    --config model \
    --output_dir "${ABS_OUT}/eval_runs/cluster_case1"

if [[ ! -f "${C1_METRICS}" ]]; then
  echo "ERROR: Missing Case1 metrics: ${C1_METRICS}" | tee -a "${STEP_LOG}" >&2
  exit 1
fi

run_step "07_case2_linear_probe_100ep" \
  "${PY}" train/train_linear_probe_case2.py --config cluster_case2_ssl20_100ep

if [[ ! -f "${C2_CKPT}" ]]; then
  echo "ERROR: Missing Case2 checkpoint: ${C2_CKPT}" | tee -a "${STEP_LOG}" >&2
  exit 1
fi

run_step "08_evaluate_case2" \
  "${PY}" eval/evaluate.py \
    --checkpoint "${C2_CKPT}" \
    --artifacts_dir "${ART_DIR}" \
    --config model \
    --output_dir "${ABS_OUT}/eval_runs/cluster_case2"

if [[ ! -f "${C2_METRICS}" ]]; then
  echo "ERROR: Missing Case2 metrics: ${C2_METRICS}" | tee -a "${STEP_LOG}" >&2
  exit 1
fi

END_TS=$(date +%s)
ELAPSED=$((END_TS - START_TS))

copy_slurm_streams_to_output

echo "" | tee -a "${STEP_LOG}"
echo "=============================================================================" | tee -a "${STEP_LOG}"
echo "SUCCESS — Unity full WISDM pipeline completed" | tee -a "${STEP_LOG}"
echo "=============================================================================" | tee -a "${STEP_LOG}"
echo "OUTPUT_ROOT (under repo):      ${OUTPUT_ROOT}" | tee -a "${STEP_LOG}"
echo "ABS_OUT:                       ${ABS_OUT}" | tee -a "${STEP_LOG}"
echo "RUNTIME_ENV:                   ${RUNTIME_ENV}" | tee -a "${STEP_LOG}"
echo "STEP_LOG:                      ${STEP_LOG}" | tee -a "${STEP_LOG}"
echo "SSL checkpoint:                ${SSL_BB}" | tee -a "${STEP_LOG}"
echo "Case1 checkpoint:              ${C1_CKPT}" | tee -a "${STEP_LOG}"
echo "Case1 metrics:                 ${C1_METRICS}" | tee -a "${STEP_LOG}"
echo "Case2 checkpoint:              ${C2_CKPT}" | tee -a "${STEP_LOG}"
echo "Case2 metrics:                 ${C2_METRICS}" | tee -a "${STEP_LOG}"
echo "Artifacts:                     ${ART_DIR}" | tee -a "${STEP_LOG}"
echo "Window cache:                  ${ABS_OUT}/cache/wisdm_windows" | tee -a "${STEP_LOG}"
echo "Total runtime (seconds):       ${ELAPSED}" | tee -a "${STEP_LOG}"
echo "=============================================================================" | tee -a "${STEP_LOG}"
