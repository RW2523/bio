# WISDM wearable HAR with AugPred SSL and a 1D spiking ResNet

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-ee4c2c.svg)](https://pytorch.org/)

Public repository: **[github.com/RW2523/bio](https://github.com/RW2523/bio)**

This repository is a **modular research codebase** for **human activity recognition (HAR)** on the **WISDM** smartphone/smartwatch inertial dataset. It implements:

1. **WISDM-only** pipelines (raw time-series; ARFF aggregates are detected but not used by default).
2. **AugPred-style self-supervised pretraining** with three binary tasks: arrow-of-time, chunk permutation, and time warping (equal loss weighting, motion-weighted sampling).
3. A **1D spiking ResNet backbone** in **pure PyTorch**: LIF neurons along time, surrogate gradients (fast-sigmoid family), temporal pooling for a fixed-size embedding.
4. **Professor-mandated linear probes** (identical splits, preprocessing, head, and metrics except backbone initialization):
   - **Case 1:** randomly initialized backbone, **fully frozen**, train linear head only.
   - **Case 2:** **AugPred-pretrained** backbone, **fully frozen**, train the same linear head only.

For **installation, UCI download/unzip, folder layout, `data_root`, troubleshooting, and Git workflow**, see **[setup.md](setup.md)** (especially **§4**).

**Compute device:** training YAML (`compute_device`) and evaluation scripts default to **CUDA** when you request `cuda` / `gpu`, for faster runs on NVIDIA hardware. Without a GPU, set `compute_device: "cpu"` in your config or pass `--device cpu` to `eval/evaluate.py` and `eval/extract_features.py`.

---

## Quick start

```bash
git clone https://github.com/RW2523/bio.git
cd bio
# See setup.md: Python venv, pip install, and WISDM data_root configuration

python3 -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
```

Point `configs/data.yaml` → `data_root` at your extracted **`wisdm-dataset`** directory (see setup.md). Then run the pipeline from the repo root.

**Requirement audit / verification log:** [VERIFY_REPORT.md](VERIFY_REPORT.md)

### Debug quickstart (smoke tests, small data)

Run this first after install to validate imports, surrogate gradients, spiking forward pass, and probe guards:

```bash
python verify/smoke_e2e.py
python data_tools/preprocess_wisdm.py --config debug
python train/pretrain_augpred.py --config debug
python train/train_linear_probe_case1.py --config debug
python train/train_linear_probe_case2.py --config debug
python eval/evaluate.py --checkpoint outputs/checkpoints/case1/best.pt --output_dir outputs/eval_runs/case1_debug
python eval/evaluate.py --checkpoint outputs/checkpoints/case2/best.pt --output_dir outputs/eval_runs/case2_debug
# CPU-only machines: add `--device cpu` to the eval lines above.
```

### Exact command order (full / production settings)

Run **in this order** (replace `<path/to/wisdm-dataset>` if your layout differs from `../wisdm-dataset` in `configs/data.yaml`):

1. **Dataset audit**  
   `python data_tools/inspect_dataset.py --data_root <path/to/wisdm-dataset> --out_dir outputs/audit`

2. **Manifest**  
   `python data_tools/build_manifest.py --data_root <path/to/wisdm-dataset> --out_dir outputs/audit`

3. **Preprocess** (window cache + splits + train-only normalization)  
   `python data_tools/preprocess_wisdm.py --config preprocess`

4. **SSL pretraining (AugPred)**  
   `python train/pretrain_augpred.py --config ssl`

5. **Case 1 — random frozen backbone + linear probe**  
   `python train/train_linear_probe_case1.py --config case1_probe`

6. **Case 2 — SSL-pretrained frozen backbone + linear probe**  
   `python train/train_linear_probe_case2.py --config case2_probe`  
   (Ensure `pretrained_backbone_path` in `configs/case2_probe.yaml` points at `outputs/checkpoints/ssl/backbone_best.pt` after step 4.)

7. **Evaluate** (writes `metrics.json`, `metrics_summary.csv`, `classification_report.txt`, `confusion_matrix.png`)  
   `python eval/evaluate.py --checkpoint outputs/checkpoints/case1/best.pt --output_dir outputs/eval_runs/case1`  
   `python eval/evaluate.py --checkpoint outputs/checkpoints/case2/best.pt --output_dir outputs/eval_runs/case2`

**Optional:** `python data_tools/report_dataset_stats.py --artifacts_dir outputs/artifacts`

### Command cheat sheet (same as above, compact)

| Step | Command |
|------|---------|
| Audit | `python data_tools/inspect_dataset.py --data_root <path/to/wisdm-dataset> --out_dir outputs/audit` |
| Manifest | `python data_tools/build_manifest.py --data_root <path/to/wisdm-dataset> --out_dir outputs/audit` |
| Preprocess (full) | `python data_tools/preprocess_wisdm.py --config preprocess` |
| Preprocess (smoke) | `python data_tools/preprocess_wisdm.py --config debug` |
| Dataset stats | `python data_tools/report_dataset_stats.py --artifacts_dir outputs/artifacts` |
| SSL pretrain | `python train/pretrain_augpred.py --config ssl` |
| Case 1 probe | `python train/train_linear_probe_case1.py --config case1_probe` |
| Case 2 probe | `python train/train_linear_probe_case2.py --config case2_probe` |
| Evaluate | `python eval/evaluate.py --checkpoint outputs/checkpoints/case1/best.pt --output_dir outputs/eval_runs/case1` (and same for case2) |
| Sequential SNN train / eval | `python train/train_wisdm_sequential_snn.py --config sequential_snn_wisdm` then `python eval/evaluate_sequential_snn.py --checkpoint outputs/checkpoints/sequential_snn/best_val_acc.pt --output_dir outputs/eval_runs/sequential_snn_test` |

---

## Obtaining WISDM

Download the official archive from the **UCI Machine Learning Repository** ([dataset 507](https://archive.ics.uci.edu/dataset/507/wisdm+smartphone+and+smartwatch+activity+and+biometrics+dataset)), unzip to a folder that contains `raw/` and `activity_key.txt`, then set `data_root` in `configs/data.yaml`. Step-by-step paths and options are in **[setup.md §4](setup.md#4-download-wisdm-and-place-it-on-disk)**.

## What this repo expects from WISDM (discovered layout)

Typical WISDM tree:

- **Raw streams (default input):** `wisdm-dataset/raw/{phone|watch}/{accel|gyro}/data_<subject>_<sensor>_<device>.txt`  
  - Rows: `subject_id,activity_code,timestamp_ns,x,y,z;`  
  - Files may be **unsorted by time**; code **sorts by timestamp** before windowing.
- **Labels:** `wisdm-dataset/activity_key.txt` (single-letter codes → activity names).
- **ARFF features (optional / not default):** `wisdm-dataset/arff_files/.../*.arff` — inventoried by audit tools only.

**Defaults (v1):** `sensor_device=phone`, `sensor_modality=accel`, `window_sec=10`, `stride_sec=5`, nominal **20 Hz** window sizing with optional per-file rate logging.

> **YAML naming:** WISDM placement uses `sensor_device` / `sensor_modality`. PyTorch device uses **`compute_device`** so merged configs never overwrite the sensor path with `"cpu"`.

---

## Repository layout

| Path | Role |
|------|------|
| `configs/` | YAML configs with `inherits` merging (`data`, `preprocess`, `model`, `ssl`, `case1_probe`, `case2_probe`, `debug`) |
| `data_tools/` | Audit, manifest, parsers, windowing, normalization, splits, preprocessing, stats reports |
| `datasets/` | `WISDMSupervisedDataset`, `WISDMSSLDataset`, weighted sampling, collate helpers |
| `transforms/` | AugPred augmentations + time warp |
| `models/` | Surrogate spike, LIF over time, spiking ResNet-1D, SSL heads, linear probe; **sequential** TR encoder, reservoir, STDP, deSNN-style head |
| `train/` | `pretrain_augpred.py`, `train_linear_probe_case1.py`, `train_linear_probe_case2.py`, `train_wisdm_sequential_snn.py`, `common.py` |
| `eval/` | Metrics, confusion matrix figure, `evaluate.py`, `evaluate_sequential_snn.py`, `analyze_sequential_snn.py`, `extract_features.py`, `visualize_embeddings.py` |
| `sbatch/` | Example Slurm driver for the sequential SNN path (`run_wisdm_sequential_snn.sbatch`) |
| `verify/` | `smoke_e2e.py` — static + small tensor checks before long runs |
| `utils/` | Seeds, logging, checkpoints, YAML merge, assertions |

Generated artifacts (`outputs/`, `.venv/`, `*.pt`, `*.npz`) are **gitignored** — reproduce locally after clone.

---

## Evaluation outputs

`eval/evaluate.py` writes:

- **`metrics.json`** — macro F1, weighted F1, accuracy, Cohen’s \(\kappa\), confusion matrix, sklearn `classification_report` dict.
- **`metrics_summary.csv`** — one row summary.
- **`confusion_matrix.png`** — plotted confusion matrix.

---

## Optional: embeddings and 2D plots

```bash
python eval/extract_features.py \
  --backbone_checkpoint outputs/checkpoints/ssl/backbone_best.pt \
  --split test \
  --artifacts_dir outputs/artifacts \
  --output outputs/features/embeddings_test.npz

python eval/visualize_embeddings.py \
  --embeddings outputs/features/embeddings_test.npz \
  --artifacts_dir outputs/artifacts \
  --output outputs/features/embedding_2d.png \
  --method umap
```

---

## Sequential reservoir SNN (WISDM) — parallel experiment path

This is a **separate** WISDM pipeline from the **1D spiking ResNet + ZCSF/identity encoder** stack in `train/snn_common.py` / `train/train_linear_probe_case1.py`. It does **not** replace or modify that path.

**Idea (high level):** continuous windows are converted to spike trains with a **threshold-based representation (TR)** on per-channel temporal differences; spikes are streamed into a **compact 3D-grid reservoir** with distance-biased (“small-world–style”) recurrence; **pair-based STDP** updates feedforward and recurrent weights on **train subjects only**; the reservoir is then **frozen**, and a **supervised readout** (`desnn` prototype head or a **linear probe** on mean firing rates) is trained with AdamW. **Subject-wise splits, window cache, and train-only normalization** are the same as the rest of the repo (`data_tools/preprocess_wisdm.py`, `datasets/wisdm_supervised_dataset.py`).

**Faithful vs approximate (vs NeuCube / deSNN literature):** TR encoding and the **ordering** (unsupervised reservoir → frozen → supervised readout) follow the paper’s methodology at a high level. The reservoir is a **practical PyTorch substitute** for NeuCube (fixed small 3D grid, Chebyshev-local + random long-range mask, LIF dynamics, dense STDP on masked weights). The readout is a **simplified deSNN-style** prototype classifier (learned class prototypes with negative squared distance), or an optional linear probe — not the full rank-order deSNN machinery.

**Run locally (after preprocess):**

```bash
# small CPU smoke (few subjects / few batches)
python train/train_wisdm_sequential_snn.py --config sequential_snn_wisdm_debug

# full-style settings (GPU recommended)
python train/train_wisdm_sequential_snn.py --config sequential_snn_wisdm

# higher-budget recipe: class-balanced loss, cosine LR, feature mixup on reservoir rates,
# dropout in the deSNN MLP, more STDP/classifier epochs, larger reservoir (see YAML)
python train/train_wisdm_sequential_snn.py --config sequential_snn_wisdm_strong

python eval/evaluate_sequential_snn.py \
  --checkpoint outputs/checkpoints/sequential_snn/best_val_acc.pt \
  --output_dir outputs/eval_runs/sequential_snn_test

python eval/analyze_sequential_snn.py \
  --checkpoint outputs/checkpoints/sequential_snn/best_val_acc.pt \
  --output_dir outputs/eval_runs/sequential_snn_analysis
```

**Alternate artifact root:** `configs/sequential_snn_wisdm_outputs2.yaml` sets `output_dir: "outputs2"` (run preprocessing with the same `output_dir` first).

**Slurm:** `sbatch/run_wisdm_sequential_snn.sbatch` uses the same Unity-style resource block as full GPU jobs, with fixed `#SBATCH --time=04:00:00` and `#SBATCH --constraint=a100` (change account/partition/mem only if your site requires it). Submit from repo root with `WISDM_DATA_ROOT` and optional `SEQUENTIAL_CONFIG` / `SEQUENTIAL_CHECKPOINT` exported.

---

## Research assumptions

- **Mixed-label windows** (activity code not constant inside a window) are **dropped by default**; counts are logged during preprocessing.
- **Subject-wise splits** — train/val/test subjects are disjoint (asserted in training scripts).
- **Normalization** — channel mean/std from **train windows only**, applied at dataset load time.
- **Dataloader tensor layout:** `[B, T, C]`; backbone input is `[B, C, T]` via `train.common.to_bct`.

---

## Citation / credits

- **WISDM dataset** and documentation: see files inside your WISDM distribution (`README.txt`, description PDF).
- **AugPred-style SSL** is inspired by self-supervised augmentation-prediction ideas in wearable HAR literature (implementations here are adapted for three binary heads and this codebase’s windowing).
- **Spiking components** follow common LIF + surrogate-gradient practice (e.g. fast-sigmoid surrogate); not a bundled third-party SNN library.

---

## License

Add a `LICENSE` file if you need a formal license for course or publication requirements; the upstream WISDM data has its own terms from the original distributors.
