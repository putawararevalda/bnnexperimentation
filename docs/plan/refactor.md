# Codebase Refactor Plan

## Target Structure

```
bnnexperimentation/
├── data/                          # (unchanged)
├── datasplit/                     # (unchanged)
├── docs/                          # (unchanged)
├── results/                       # all results consolidated here
│   ├── shipsnet/
│   │   ├── bayesian/
│   │   ├── deterministic/
│   │   └── seu/
│   └── eurosat/
│       ├── bayesian/
│       └── seu/
├── src/
│   ├── models/
│   │   ├── __init__.py
│   │   ├── bayesian_cnn.py        # BayesShipsCNN (canonical, from shipsnet_newslate.py)
│   │   ├── deterministic_cnn.py   # deterministic CNN baseline
│   │   └── components.py          # SmartPool, custom activations (WG, RWG)
│   ├── data/
│   │   ├── __init__.py
│   │   ├── shipsnet.py            # ShipsNet load_data()
│   │   └── eurosat.py             # EuroSAT load_data()
│   ├── training/
│   │   ├── __init__.py
│   │   └── svi.py                 # train_svi_with_stats(), plot helpers
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py             # AAD, Softmax Difference, ARIn
│   │   └── seu.py                 # bitflip.py logic moved here
│   └── utils/
│       ├── __init__.py
│       ├── notify.py              # send_telegram_message()
│       └── guide.py               # AutoDiagonalLaplace (from utils/guide.py)
├── scripts/
│   ├── train_shipsnet.py          # replaces shipsnet_newslate_guide_project.py
│   ├── train_eurosat.py           # replaces gputrain.py
│   ├── eval_seu_shipsnet.py       # replaces shipsnet-experiment05-guide.py
│   ├── eval_seu_deterministic.py  # replaces shipsnet-experiment05-deterministic00_custom.py
│   └── train_deterministic.py     # replaces shipsnet-train-deterministic.py
├── notebooks/
│   ├── exploration/               # EDA notebooks
│   └── experiments/               # key experiment notebooks (curated)
├── archive/                       # old/superseded files (not deleted)
├── tests/
│   └── smoke_test.py              # import check + one forward pass
├── CLAUDE.md
├── .env
├── requirements.txt
└── environment.yml
```

---

## Phase 1 — Consolidate `src/` ✅ DONE

- [x] Create `src/` directory with the subdirectories above
- [x] **`src/models/bayesian_cnn.py`** — Move `BayesShipsCNN` from `shipsnet_newslate.py`; this is the canonical model used in the paper. Delete inline model definitions in all other scripts.
- [x] **`src/models/deterministic_cnn.py`** — Extract deterministic CNN class from `shipsnet-train-deterministic.py`
- [x] **`src/models/components.py`** — Move `SmartPool` (from `shipsnet_newslate_guide_project.py`), `WeightedGaussian`, `WeightedGaussianActivation` (from `utils/function.py`)
- [x] **`src/data/shipsnet.py`** — Move `load_data()` from `shipsnet_newslate.py`
- [x] **`src/data/eurosat.py`** — Move `load_data()` from `utils/function.py`
- [x] **`src/training/svi.py`** — Move `train_svi_with_stats()` and `plot_training_results_with_stats()` from `shipsnet_newslate.py` (most complete version)
- [x] **`src/evaluation/seu.py`** — Move `bitflip_float32()`, `bitflip_float32_with_original()`, `float32_to_binary()`, `binary_to_float32()` from `bitflip.py`
- [x] **`src/evaluation/metrics.py`** — Extract AAD, Softmax Difference, ARIn calculation logic into standalone functions
- [x] **`src/utils/notify.py`** — Move `send_telegram_message()` from `utils/function.py`
- [x] **`src/utils/guide.py`** — Move `AutoDiagonalLaplace` from `utils/guide.py`; also added `AutoUniform`
- [ ] Delete old `utils/` directory after migration (or keep as shim that re-exports from `src/`)

## Phase 2 — Consolidate `scripts/` ✅ DONE

- [x] Create `scripts/` directory
- [x] **`scripts/train_shipsnet.py`** — Clean version of `shipsnet_newslate_guide_project.py`; update all imports to use `src.*`; remove inline class definitions
- [x] **`scripts/train_eurosat.py`** — Clean version of `gputrain.py`; update imports to use `src.*`
- [x] **`scripts/eval_seu_shipsnet.py`** — Clean version of `shipsnet-experiment05-guide.py`; update imports; `NewInjector` kept in script, `pyro_param_store_path` now passed as constructor arg
- [x] **`scripts/eval_seu_deterministic.py`** — Clean version of `shipsnet-experiment05-deterministic00_custom.py`; update imports
- [x] **`scripts/train_deterministic.py`** — Clean version of `shipsnet-train-deterministic.py`; update imports

## Phase 3 — Archive Superseded Files ✅ DONE

Move to `archive/` (do not delete — they may contain useful experiment-specific tweaks):

- [x] `eurosat_newslate.py`, `eurosat_newslate_B.py`, `eurosat_newslate_C.py` (duplicates)
- [x] `shipsnet_newslate_guide_project_old.py`
- [x] `shipsnet_newslate_guide_project_uniform_test.py`
- [x] `shipsnet-experiment03.py`, `shipsnet-experiment04.py` (superseded by 05)
- [x] `shipsnet-experiment05-guide-median.py`, `shipsnet-experiment05-guide-remedy0.py` (one-off variants)
- [x] `shipsnet_train.py`, `shipsnet_train-guide.py` (early versions)
- [x] `eurosat.py`, `eurosat_cnn.py`, `eurosat_bayesian_cnn.py`, `eurosat_bcnn_simple_training.py` (pre-newslate)
- [x] `all_GP_train.py` (absorbed into train_eurosat.py)
- [x] `vi.py`, `pyrotest.py`, `pyrotest_tutorial.py`, `sample-utils.py`, `script.py`, `smalltraintest.py` (scratch/tutorial files)
- [x] `shipsnet_newslate_guide_scale.py`, `shipsnet_newslate_multivariate_project.py` (experimental variants)
- [x] `datasplit_eurosat_cnn.py`, `datasplit_eurosat_bcnn.py` (one-time setup scripts)
- [x] Original root-level `bitflip.py`, `custoptimizer.py`
- [x] `shipsnet-train-deterministic.py` (superseded by scripts/train_deterministic.py)
- [x] `shipsnet_newslate.py`, `shipsnet_newslate_guide.py`, `shipsnet_newslate_guide_project.py`, `shipsnet_newslate_guide_project_mvrt.py` (superseded by scripts/train_shipsnet.py)
- [x] `shipsnet-experiment05-guide.py` (superseded by scripts/eval_seu_shipsnet.py)
- [x] `shipsnet-experiment05-deterministic00_custom.py` (superseded by scripts/eval_seu_deterministic.py)
- [x] `gputrain.py` (superseded by scripts/train_eurosat.py)
- [x] `shipsnet-check_nan.ipynb` — NOT moved (notebook, handled by Phase 4)
- [x] `shipsnet_newslate_multivariate_project.py` — not found at root (already archived above)

## Phase 4 — Organize Notebooks ✅ DONE

- [x] Create `notebooks/exploration/` — Move EDA notebooks: `shipsnet-check_nan.ipynb`, `shipsnet-uniform-inspection.ipynb`, `shipsnet-uniform-guide-check.ipynb`, `delete_files.ipynb`, `check_completeness.ipynb`
- [x] Create `notebooks/experiments/` — Move the key experiment notebooks that produced paper results (curate: keep only one representative per activation/prior combination):
  - `eurosat_bcnn_simple_gpu_gaussian_random.ipynb`
  - `eurosat_bcnn_simple_gpu_laplace_random.ipynb`
  - `shipsnet_newslate.ipynb`
  - `shipsnet_newslate_guide.ipynb` (not found — skipped)
  - `shipsnet-experiment04.ipynb`
- [x] Archive remaining `eurosat_bcnn_simple_gpu_*.ipynb` notebooks (keep only the canonical ones above) — move to `archive/notebooks/`
- [x] Update any hardcoded paths in kept notebooks to reflect new `src/` structure

## Phase 5 — Consolidate Results ✅ DONE

- [x] Create `results/shipsnet/bayesian/`, `results/shipsnet/deterministic/`, `results/shipsnet/seu/`
- [x] Create `results/eurosat/bayesian/`, `results/eurosat/seu/`
- [x] Move existing results directories into the new structure:
  - `results_shipsnet_v02_*/` → `results/shipsnet/bayesian/`
  - `results_shipsnet_deterministic_*/` → `results/shipsnet/deterministic/`
  - `results_shipsnet_*_SEU*/` → `results/shipsnet/seu/`
  - `results_GP_eurosat*/`, `results_eurosat/` → `results/eurosat/bayesian/`
- [x] Update `save_dir` default paths in all scripts to match new structure

## Phase 6 — Smoke Test ✅ DONE

- [x] Create `tests/smoke_test.py`:
  - Import all `src.*` modules successfully
  - Run one forward pass through `BayesShipsCNN` with a random tensor `(2, 3, 64, 64)`
  - Run one forward pass through the deterministic CNN
  - Call `bitflip_float32(1.5, bit_i=5)` and verify output is a float
  - Verify `ARIn` computation on dummy values
- [x] Run: `uv run python tests/smoke_test.py` — 6/6 tests passed
- [x] Run: `uv run python scripts/train_shipsnet.py --help` — exits cleanly
- [x] Run: `uv run python scripts/eval_seu_shipsnet.py --help` — exits cleanly

---

## Files to Keep In-Place (root level)

- `CLAUDE.md`, `README.md`, `.env`, `.gitignore`
- `requirements.txt`, `environment.yml`, `pytorch-requirements.txt`
- `install.sh`, `download_shipsnet.sh`
- `shipsnet_newslate_guide_project.sh` (runner script — update paths after refactor)
- `data/`, `datasplit/`, `docs/`
