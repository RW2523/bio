# Setup guide — WISDM HAR + AugPred + Spiking ResNet

This document covers **environment setup**, **WISDM data placement**, **configuration**, **verification**, **Git/GitHub**, and **common failures**.

**Repository:** [https://github.com/RW2523/bio](https://github.com/RW2523/bio)

---

## 1. Prerequisites

| Requirement | Notes |
|---------------|--------|
| **Python** | 3.10+ recommended (3.11/3.12/3.14 tested in development). |
| **Git** | For clone/push. |
| **Disk** | Raw WISDM + cached windows + checkpoints can require **several GB** for full runs. |
| **RAM** | Full preprocessing loads per-subject streams; 8 GB+ comfortable for default settings. |
| **GPU** | **Recommended for training.** Training and eval configs default to `compute_device: "cuda"` (and eval CLIs default to `--device cuda`). Use `compute_device: "cpu"` in YAML or `--device cpu` for eval if you have no CUDA-capable GPU or a CPU-only PyTorch build. |

**macOS / Linux (Homebrew Python):** PEP 668 may block global `pip install`. **Always use a virtual environment** (below).

---

## 2. Clone the repository

```bash
git clone https://github.com/RW2523/bio.git
cd bio
```

If you use SSH remotes:

```bash
git clone git@github.com:RW2523/bio.git
cd bio
```

---

## 3. Python virtual environment and dependencies

From the repository root (`bio/`):

```bash
python3 -m venv .venv
source .venv/bin/activate          # Linux/macOS
# OR:  .venv\Scripts\activate     # Windows PowerShell/CMD
```

Upgrade pip (optional but recommended):

```bash
python -m pip install --upgrade pip
```

Install packages:

```bash
python -m pip install -r requirements.txt
```

**Core stack:** PyTorch, NumPy, PyYAML, scikit-learn, matplotlib, tqdm, pandas; **optional:** `umap-learn` for `eval/visualize_embeddings.py` (UMAP path).

**PyTorch with CUDA (optional):** install the wheel appropriate for your CUDA version from [pytorch.org](https://pytorch.org/get-started/locally/), then reinstall other deps from `requirements.txt` if needed.

---

## 4. Download WISDM and place it on disk

This repository **does not ship** the WISDM files. Download them from the **UCI Machine Learning Repository** (official host for this release):

**Dataset page (ID 507):** [WISDM Smartphone and Smartwatch Activity and Biometrics Dataset](https://archive.ics.uci.edu/dataset/507/wisdm+smartphone+and+smartwatch+activity+and+biometrics+dataset)

**Citation (use in papers / reports):** Weiss, G. (2019). *WISDM Smartphone and Smartwatch Activity and Biometrics Dataset* [Dataset]. UCI Machine Learning Repository. [https://doi.org/10.24432/C5HK59](https://doi.org/10.24432/C5HK59).  
License: **CC BY 4.0** (see UCI page for details).

### 4.1 Download the archive

On the UCI page above, use **Download** to get **`wisdm-dataset.zip`** (~295 MB). That zip is what this project expects (it contains `raw/`, `activity_key.txt`, ARFF exports, etc.).

> **Note:** UCI also documents `pip install ucimlrepo` and a Python `fetch_ucirepo(id=507)` API. That API is convenient for tabular views of the data, but **this codebase is built around the on-disk layout produced by unzipping `wisdm-dataset.zip`**. For training and preprocessing here, **download and unzip the zip** as below.

### 4.2 Unzip and choose a folder layout

After download, unzip **`wisdm-dataset.zip`**. You should get a directory named **`wisdm-dataset`** (or similar) whose **immediate** children include:

- `activity_key.txt`
- `raw/` (with `phone/`, `watch/`, etc.)
- optionally `arff_files/`, PDFs, etc.

**Sanity check:** this path must be the directory that **directly** contains `raw/`:

```text
<DATA_ROOT>/activity_key.txt
<DATA_ROOT>/raw/phone/accel/data_1600_accel_phone.txt   # example
```

If your unzip tool created an extra top-level folder, move the inner `wisdm-dataset` so `DATA_ROOT/raw/...` exists as above.

### 4.3 Where to put `wisdm-dataset` (recommended)

**Option A — Sibling folder next to the repo (recommended)**  
Keeps large data **outside** the git repo and matches the default config.

```text
your-workspace/
  bio/                    # this repository (clone of RW2523/bio)
  wisdm-dataset/          # unzipped UCI archive (NOT committed to git)
    activity_key.txt
    raw/
    ...
```

Then in **`configs/data.yaml`** (default):

```yaml
data_root: "../wisdm-dataset"
```

From inside `bio/`, `../wisdm-dataset` resolves to the sibling folder.

**Option B — Fixed location on your machine**

Put `wisdm-dataset` anywhere, e.g.:

```text
/Users/you/Datasets/wisdm-dataset/
```

Set an **absolute** path in `configs/data.yaml`:

```yaml
data_root: "/Users/you/Datasets/wisdm-dataset"
```

**Option C — Inside the repo (optional)**

```text
bio/
  external/
    wisdm-dataset/
      activity_key.txt
      raw/
      ...
```

```yaml
data_root: "external/wisdm-dataset"
```

Add `external/` to **`.gitignore`** if you do this, so you do not accidentally commit gigabytes of sensor data.

### 4.4 Point the code at your `DATA_ROOT`

Edit **`configs/data.yaml`**:

```yaml
data_root: "../wisdm-dataset"   # or absolute path; must contain raw/ + activity_key.txt
sensor_device: "phone"         # phone | watch
sensor_modality: "accel"       # accel | gyro
```

- **`data_root`** is resolved **relative to the repository root** unless the path is absolute (`train/common.resolve_path` / `utils.paths`).
- **`sensor_device` / `sensor_modality`** select which raw stream under `raw/<device>/<modality>/` is used.

### 4.5 Verify the install

From the repo root, with venv activated:

```bash
python data_tools/inspect_dataset.py --data_root ../wisdm-dataset --out_dir outputs/audit
# If you used Option B/C, replace ../wisdm-dataset with your DATA_ROOT path
```

Open **`outputs/audit/DATASET_AUDIT_REPORT.txt`** and confirm raw file counts under `raw/phone/accel` (etc.) look reasonable. Then:

```bash
python data_tools/build_manifest.py --data_root ../wisdm-dataset --out_dir outputs/audit
```

If either command errors with “missing `raw/`” or “no `data_*.txt` files”, your **`data_root` is one level too high or too low** — adjust until `DATA_ROOT/raw/phone/accel` exists.

### 4.6 Expected directory layout (reference)

```
wisdm-dataset/
  activity_key.txt
  README.txt
  WISDM-dataset-description.pdf
  raw/
    phone/
      accel/*.txt
      gyro/*.txt
    watch/
      accel/*.txt
      gyro/*.txt
  arff_files/          # present in UCI bundle; optional for this repo’s default pipeline
    phone/
    watch/
```

---

## 5. YAML configuration overview

| File | Purpose |
|------|---------|
| `configs/data.yaml` | WISDM root path, sensor stream selection, optional `max_subjects` / `debug_subjects` |
| `configs/preprocess.yaml` | Windowing, stride, nominal Hz, splits, cache paths (`inherits: data.yaml`) |
| `configs/model.yaml` | Backbone width/depth, LIF/surrogate hyperparameters (`inherits` where referenced) |
| `configs/ssl.yaml` | SSL epochs, LR, AugPred hyperparameters, `compute_device` |
| `configs/case1_probe.yaml` | Random frozen backbone + probe; `compute_device`, checkpoint dir |
| `configs/case2_probe.yaml` | Same, but **`pretrained_backbone_path`** → `outputs/checkpoints/ssl/backbone_best.pt` (adjust after SSL run) |
| `configs/debug.yaml` | Small subject count + 1 epoch smoke (`inherits` multiple files) |

**Merging:** `utils/yaml_config.py` loads `inherits` chains and deep-merges; child keys override parents.

---

## 6. End-to-end workflow (recommended order)

1. **Audit** — confirm paths, modalities, inferred timing on a sample file.  
2. **Manifest** — machine-readable inventory of raw streams.  
3. **Preprocess** — build `outputs/cache/wisdm_windows/*.npz` and `outputs/artifacts/*.json`.  
4. **SSL** — `train/pretrain_augpred.py` → saves `backbone_best.pt` among others.  
5. **Case 1** — random frozen backbone + linear head.  
6. **Case 2** — load SSL backbone, freeze, train same head.  
7. **Evaluate** — test metrics + confusion matrix.

Exact commands are summarized in **[README.md](README.md)**.

---

## 7. Smoke test (fast)

Uses three subjects and one epoch:

```bash
python data_tools/preprocess_wisdm.py --config debug
python train/pretrain_augpred.py --config debug
python train/train_linear_probe_case1.py --config debug
python train/train_linear_probe_case2.py --config debug
python eval/evaluate.py --checkpoint outputs/checkpoints/case1/best.pt --output_dir outputs/eval_runs/case1_debug
```

`configs/debug.yaml` sets `pretrained_backbone_path` so Case 2 can find SSL weights without editing `case2_probe.yaml`.

---

## 8. Git and GitHub

### 8.1 What is tracked

`.gitignore` excludes:

- `outputs/` (logs, caches, checkpoints, figures)
- `.venv/`, `venv/`
- `*.pt`, `*.pth`, `*.npz`
- `__pycache__/`, common tool caches

**Do not** force-add WISDM raw data or large checkpoints unless you intend to use Git LFS and have rights to redistribute.

### 8.2 First-time push (maintainers)

```bash
cd bio
git init
git branch -M main
git remote add origin https://github.com/RW2523/bio.git
git add -A
git status   # verify no huge binaries
git commit -m "Initial import: WISDM HAR, AugPred SSL, spiking ResNet-1D, linear probes"
git push -u origin main
```

If the GitHub repo already has a commit (e.g. README created on the website), use:

```bash
git pull origin main --rebase
git push -u origin main
```

### 8.3 Authentication

- **HTTPS:** GitHub [personal access token](https://docs.github.com/en/authentication) as password, or Git Credential Manager.
- **SSH:** `git remote set-url origin git@github.com:RW2523/bio.git` and ensure `ssh -T git@github.com` works.

### 8.4 GitHub CLI (optional)

```bash
gh auth login
gh repo clone RW2523/bio
```

---

## 9. Troubleshooting

| Symptom | Likely cause | Fix |
|--------|----------------|-----|
| `ModuleNotFoundError: yaml` | No venv / deps not installed | Activate `.venv` and `pip install -r requirements.txt` |
| `FileNotFoundError` for `raw/.../accel` | Wrong `data_root` or wrong `sensor_device` / `sensor_modality` | Fix `configs/data.yaml`; rerun audit |
| “Missing `raw/`” but zip is extracted | **Extra** top-level folder after unzip (`wisdm-dataset/wisdm-dataset/raw`) | Point `data_root` at the **inner** folder that directly contains `raw/` and `activity_key.txt` (see §4.2) |
| `Empty test split` / split errors | Too few subjects after `max_subjects` / `debug_subjects` | Increase subjects or relax split logic in `data_tools/splits.py` |
| SSL Case 2 cannot load weights | Missing SSL run or wrong `pretrained_backbone_path` | Run SSL first; path must point to `backbone_best.pt` or compatible checkpoint |
| Very slow training | CPU-only run on large data | Ensure NVIDIA drivers + CUDA PyTorch; keep `compute_device: "cuda"`, reduce `batch_size`, or use `debug` config |
| `RuntimeError` about `torch.cuda.is_available()` is False | No GPU / CPU-only PyTorch while defaults request CUDA | Install CUDA-enabled PyTorch with a visible GPU, or set `compute_device: "cpu"` in the YAML (and `python eval/evaluate.py ... --device cpu`) |
| `externally-managed-environment` (pip) | System Python on macOS | Always use a venv (section 3) |
| UMAP errors | `umap-learn` not installed / incompatible | `pip install umap-learn` or use `--method tsne` in `visualize_embeddings.py` |

---

## 10. Updating this documentation

After changing defaults or scripts:

1. Update **README.md** (commands + high-level design).  
2. Update **setup.md** (environment, paths, Git, troubleshooting).  
3. Run the **debug** pipeline once to confirm nothing regressed.

---

## 11. Support

For course or research questions, open an **Issue** on [RW2523/bio](https://github.com/RW2523/bio) with your OS, Python version, config name, and the **first error traceback** plus the **smallest command** that reproduces it.
