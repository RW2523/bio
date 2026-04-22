#!/bin/bash
# =============================================================================
# UMass Unity — FULL Case 1 pipeline (all subjects, preprocess + probe + eval)
#
# This is NOT the short gpu-preempt smoke script. It uses:
#   - partition `gpu` (non-preempt, up to 48h default max on Unity general GPU)
#   - NO `--qos=short` (that QoS is for shorter jobs; full WISDM often needs many hours)
#   - A100 via `--constraint=a100`
#   - Explicit merges: `preprocess` + `case1_probe` (never `debug.yaml`)
#
# Docs: https://docs.unity.rc.umass.edu/documentation/tools/gpus/
#       https://docs.unity.rc.umass.edu/documentation/jobs/sbatch
#
# Submit (from anywhere, if you set absolute paths):
#   export PROJECT_ROOT=/path/to/project    # contains train/, configs/, data_tools/
#   export WISDM_DATA_ROOT=/path/to/wisdm-dataset
#   sbatch cluster/unity_case1_full_a100.sh
#
# Or from repo root:
#   cd /path/to/project && export WISDM_DATA_ROOT=/work/.../wisdm-dataset
#   sbatch cluster/unity_case1_full_a100.sh
#
# Override walltime / memory if Unity rejects the request:
#   sbatch --time=12:00:00 --mem=256G cluster/unity_case1_full_a100.sh
# =============================================================================

#SBATCH --job-name=wisdm-case1-full
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00
#SBATCH --gpus=1
#SBATCH --constraint=a100
#SBATCH --output=slurm-case1-full-%j.out
#SBATCH --error=slurm-case1-full-%j.err

set -euo pipefail

# ------------- Required paths ------------------------------------------------
if [[ -z "${WISDM_DATA_ROOT:-}" ]]; then
  echo "ERROR: export WISDM_DATA_ROOT=/path/to/wisdm-dataset (folder with raw/ and activity_key.txt)." >&2
  exit 1
fi

: "${PROJECT_ROOT:=${SLURM_SUBMIT_DIR:-}}"
if [[ ! -f "${PROJECT_ROOT}/train/train_linear_probe_case1.py" ]]; then
  echo "ERROR: PROJECT_ROOT must point at the repo root (has train/). Got: ${PROJECT_ROOT}" >&2
  echo "Fix: export PROJECT_ROOT=/absolute/path/to/project before sbatch." >&2
  exit 1
fi

: "${VENV_ROOT:=${PROJECT_ROOT}/.venv}"

# ---------------------------------------------------------------------------
cd "${PROJECT_ROOT}"
mkdir -p outputs/audit outputs/eval_runs/case1

# shellcheck source=/dev/null
source "${VENV_ROOT}/bin/activate"

# Optional: match your PyTorch CUDA build (module spider cuda)
# module load cuda/12.6
# module load cudnn/8.9.7.29-12-cuda12.6

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"

# --- Patch data.yaml: WISDM path + force full cohort (never debug caps) ----
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

# --- Verify merged preprocess config does not inherit debug limits ----------
python3 <<'VERIFY' || exit 1
import os
import sys
from pathlib import Path

pr = Path(os.environ["PROJECT_ROOT"])
sys.path.insert(0, str(pr))
os.chdir(pr)
from utils.yaml_config import load_merged_config

cfg = load_merged_config("preprocess")
ms = cfg.get("max_subjects")
ds = cfg.get("debug_subjects") or []
if ms is not None:
    raise SystemExit(f"Refusing to run: merged preprocess has max_subjects={ms!r} (expected null for full run).")
if cfg.get("debug_subjects"):
    raise SystemExit(f"Refusing to run: merged preprocess has debug_subjects={cfg.get('debug_subjects')!r}.")
print("VERIFY OK: merged `preprocess` uses all subjects (max_subjects is null).")

c1 = load_merged_config("case1_probe")
ep = int(c1.get("epochs", 0))
if ep < 2:
    print(f"WARNING: case1_probe epochs={ep} is very low for a full study.", file=sys.stderr)
else:
    print(f"VERIFY OK: case1_probe epochs={ep}.")
VERIFY

echo "=== GPU ==="
nvidia-smi -L || true

OUT_AUDIT="${PROJECT_ROOT}/outputs/audit"

echo "=== 1) Dataset audit ==="
python data_tools/inspect_dataset.py --data_root "${WISDM_DATA_ROOT}" --out_dir "${OUT_AUDIT}"

echo "=== 2) Manifest ==="
python data_tools/build_manifest.py --data_root "${WISDM_DATA_ROOT}" --out_dir "${OUT_AUDIT}"

echo "=== 3) Preprocess FULL (configs/preprocess + data; all subjects) ==="
python data_tools/preprocess_wisdm.py --config preprocess

echo "=== 4) Case 1 linear probe (configs/case1_probe; frozen random backbone) ==="
python train/train_linear_probe_case1.py --config case1_probe

echo "=== 5) Evaluate Case 1 on held-out test ==="
python eval/evaluate.py \
  --checkpoint "${PROJECT_ROOT}/outputs/checkpoints/case1/best.pt" \
  --output_dir "${PROJECT_ROOT}/outputs/eval_runs/case1"

echo "=== Done ==="
echo "Checkpoints: ${PROJECT_ROOT}/outputs/checkpoints/case1/"
echo "Metrics:     ${PROJECT_ROOT}/outputs/eval_runs/case1/metrics.json"
