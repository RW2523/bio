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
| **GPU** | Optional. All configs default to `compute_device: "cpu"`; set to `cuda` if PyTorch sees a GPU. |

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

## 4. WISDM dataset placement

This repository **does not redistribute** WISDM raw files. You must obtain the dataset separately and point the code at it.

### 4.1 Expected directory layout

The code expects a folder (commonly named `wisdm-dataset`) containing at least:

```
wisdm-dataset/
  activity_key.txt
  raw/
    phone/
      accel/*.txt
      gyro/*.txt
    watch/
      accel/*.txt
      gyro/*.txt
  arff_files/          # optional for your audit; not used by default training pipeline
    ...
```

### 4.2 Configure `data_root`

Edit **`configs/data.yaml`**:

```yaml
data_root: "../wisdm-dataset"   # relative to repo root, OR use an absolute path
sensor_device: "phone"         # phone | watch
sensor_modality: "accel"       # accel | gyro
```

- **`data_root`** must resolve to the folder that **contains** `raw/` and `activity_key.txt`.
- Paths in configs are resolved **relative to the repository root** when not absolute (see `train/common.resolve_path` / `utils.paths`).

**Tip:** If your tree differs, run the audit script with an explicit `--data_root` first:

```bash
python data_tools/inspect_dataset.py --data_root /path/to/wisdm-dataset --out_dir outputs/audit
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
| `Empty test split` / split errors | Too few subjects after `max_subjects` / `debug_subjects` | Increase subjects or relax split logic in `data_tools/splits.py` |
| SSL Case 2 cannot load weights | Missing SSL run or wrong `pretrained_backbone_path` | Run SSL first; path must point to `backbone_best.pt` or compatible checkpoint |
| Very slow training | `compute_device: cpu` on large data | Use GPU, reduce `batch_size`, or use `debug` config |
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
