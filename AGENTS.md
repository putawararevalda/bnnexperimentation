# AGENTS.md

This document serves as the operational guide and knowledge base for AI coding agents (Antigravity, Claude Code, Cursor, Codex, etc.) collaborating on this repository.

---

## 1. Research Context & Objectives

This repository contains the research codebase for investigating the resilience of **Bayesian Neural Networks (BNNs)** against **Single Event Upsets (SEUs / FP32 bitflips)** in mission-critical and radiation-harsh environments (e.g., aerospace, satellite computing).

- **Primary Paper**: `docs/2026_ICAART_Revalda (14).pdf`
- **MSc Dissertation**: `docs/dissertation/PutawaraRevalda_mscthesis (24).pdf`
- **Benchmark Datasets**:
  - **ShipsNet**: Binary classification (ship / no-ship) from satellite tiles.
  - **EuroSAT**: 10-class multispectral/RGB land-cover satellite imagery (with additional 2-class binary investigations under `results/eurosat_binary/`).

---

## 2. Governing Plans & Statistical Rules

### 2.1 The Master Plan
**`docs/revision/results_presentation_plan.html`** is the governing authority for paper revision, presentation strategy, page budgets, and statistical reporting. Agents must consult this document before altering tables, figures, or analytical methodology.

Supporting documentation:
- Reviewer checklist: `docs/revision/reviewer_feedback.md`
- Original reviews: `docs/revision/Reviews_ICAART_2026_90.pdf` and `ICAARTreview/`

### 2.2 Fixed Methodological Decisions
Do not alter these statistical and reporting constraints without explicit user request:
1. **Confidence Interval (CI) Unit of Analysis = The Fold**:
   - $n = 5$, Student-$t$ distribution with $t(0.975, 4) = 2.776$.
   - **Never** compute sample CIs over the ~336 injection rows (which reflects injection-site variance, not model training variance).
2. **Paired Per-Fold Differences**:
   - Any two-condition claims (BNN vs DNN, Weighted Gaussian vs ReLU) must evaluate paired differences per fold. Shared fold variance cancels in the difference.
3. **Fixed Page Budget**:
   - Space additions must be compensated by condensation/deletion (e.g., combining EuroSAT and ShipsNet into unified comparison tables and `layer × bit` heatmaps).
4. **EuroSAT Primary Findings**:
   - Activation function is the dominant factor (Weighted Gaussian outperforms ReLU by ~46% on ARIn and +7% accuracy).
   - Architectural regularizers (base, smartpool, dropout, weight decay) produce a null result on EuroSAT (ARIn 0.0969–0.0983) and require only concise textual notation.
5. **Double-Softmax Metric Floor**:
   - Follow `docs/analysis/seu_metric_noise_floor.md`. Ensure DNN baseline evaluation avoids double-softmax application before computing metric comparisons.

---

## 3. Environment & Execution Setup

All Python workflows use [`uv`](https://docs.astral.sh/uv/) for environment and package execution.

### 3.1 Common Commands

```bash
# Run any script with uv
uv run python <script_path>.py

# Run with temporary dependencies
uv run --with pyro-ppl,torch,torchvision,tqdm,scikit-learn,python-dotenv,requests,matplotlib,pandas,numpy python <script_path>.py

# Install project dependencies
uv pip install -r requirements.txt

# Run the test suite / smoke test
uv run --with pyro-ppl,torch,torchvision,tqdm,scikit-learn,python-dotenv,requests,matplotlib,pandas,numpy python tests/smoke_test.py
uv run pytest tests/
```

### 3.2 GPU / CUDA Support
GPU packages (CUDA 12.8: `torch`, `torchvision`, `torchaudio`) are specified in `requirements.txt` and `environment.yml`. Install matching CUDA wheels if performing full GPU training/sweeps.

---

## 4. Codebase Architecture (`src/`)

All modular, reusable library code resides under `src/` and must be imported via absolute package paths (`from src.<module> import ...`).

```
src/
├── models/
│   ├── bayesian_cnn.py      # BayesShipsCNN — canonical BNN (priors, activations, smartpool, dropout)
│   ├── deterministic_cnn.py # ShipsCNNCustom — standard deterministic CNN baseline
│   ├── components.py        # SmartPool, WeightedGaussian, WeightedGaussianActivation, UniformReal
│   └── components_fast.py   # SmartPoolFast (opt-in fast pooling implementation)
├── data/
│   ├── shipsnet.py          # load_data(), load_data_withval()
│   ├── eurosat.py           # load_data() for 10-class EuroSAT
│   ├── eurosat_binary.py    # Binary EuroSAT loader
│   └── folds.py             # K-fold cross validation split helpers
├── training/
│   └── svi.py               # train_svi_with_stats(), predict_data(), plot_training_results_with_stats()
├── evaluation/
│   ├── seu.py               # bitflip_float32(), bitflip_float32_with_original()
│   ├── metrics.py           # absolute_accuracy_difference(), softmax_difference(), aggregate_robustness_index()
│   └── aggregate.py         # Result aggregation, CSV parsers, summary tables
└── utils/
    ├── guide.py             # AutoLaplace, AutoUniform (custom Pyro variational guides)
    └── notify.py            # Telegram dispatch notifications (send_telegram_message)
```

### 4.1 Pyro SVI Model-Guide Pattern
All Bayesian neural network training follows Pyro's Stochastic Variational Inference (SVI) paradigm:
- **Model**: `BayesShipsCNN` (`PyroModule`) with parameter priors instantiated via `_make_prior()`.
- **Variational Guides**: `AutoNormal` (Gaussian), `AutoLaplace`, or `AutoUniform`.
- **Loss / Optimization**: `SVI` with `pyro.infer.Trace_ELBO()` and Adam optimizer.
- **Inference**: Monte Carlo sampling ($S=10$) via `pyro.poutine.trace(guide)` and `pyro.poutine.replay(model)`, followed by averaging predictive logits before `argmax` (`src/training/svi.py:predict_data`).
- **Param Store**: Variational parameters reside in Pyro's global parameter store (`pyro.get_param_store()`). Always manage or reset state with `pyro.clear_param_store()` when switching checkpoints.

### 4.2 SEU Injection (`src/evaluation/seu.py`)
- Emulates single-event upsets via bit-level manipulation of 32-bit floating point weights (`bitflip_float32`).
- Target indexing adheres to IEEE-754 FP32:
  - Bit 0: Sign bit
  - Bits 1–8: Exponent bits (bits 1, 3, 6 tested)
  - Bits 9–31: Mantissa bits (bits 10, 15, 21 tested)

### 4.3 Robustness Metrics (`src/evaluation/metrics.py`)
- **AAD (Absolute Accuracy Difference)**:
  $$\text{AAD} = |\text{Accuracy}_{\text{post-SEU}} - \text{Accuracy}_{\text{pre-SEU}}|$$
- **Softmax Difference**:
  $$\text{SoftmaxDiff} = \frac{1}{N} \sum_{i=1}^N \|\text{Softmax}(\mathbf{z}_i^{\text{post}}) - \text{Softmax}(\mathbf{z}_i^{\text{pre}})\|_\infty$$
- **ARIn (Aggregate Robustness Index)** (lower indicates greater robustness):
  $$\text{ARIn} = \sqrt{\frac{\text{AAD}^2 + \text{SoftmaxDiff}^2}{2}}$$

---

## 5. Experimental Grid & Variables

| Variable | Values / Options |
|---|---|
| **Prior Distributions** | `gaussian`, `laplace`, `uniform` |
| **Prior Scales ($b$)** | `10.0`, `1.0`, `0.1` |
| **Activation Functions** | `relu`, `tanh`, `sigmoid`, `sinusoidal`, `relu6`, `wg` (Weighted Gaussian), `rwg` (Relu + Weighted Gaussian) |
| **Architectural Variants** | `v02_00`: Base CNN<br>`v02_01`: SmartPool<br>`v02_02`: Dropout ($p=0.5$)<br>`v02_03`: Weight Decay ($\lambda=10^{-4}$) |
| **SEU Injection Targets** | **Layers**: `conv1`, `conv2`, `fc1`<br>**Parameters**: `weight`, `bias`<br>**Bit Positions**: `0`, `1`, `3`, `6`, `10`, `15`, `21` |

---

## 6. Directory Layout & Results Organization (`results/`)

### 6.1 ShipsNet Layout
- **Fold 1 (Paper Baseline)**: Located directly at the root results folders (not under `fold1/`):
  - BNN models: `results/shipsnet/bayesian/results_shipsnet_v02_{00,01,02,03}/`
  - Deterministic DNN: `results/shipsnet/deterministic/results_shipsnet_deterministic_{00..03}[_SEU]/`
  - SEU evaluations: `results/shipsnet/seu/results_shipsnet_v02_{00..03}_SEU/`
- **Folds 1–5 (Cross-Validation)**:
  - BNN Training: `results/shipsnet/bayesian/fold{1..5}/{base,dropout,smartpool,weight_decay}/` (21 models each, complete).
  - BNN SEU evaluations: `results/shipsnet/seu/fold{1..5}/{base,dropout,smartpool,weight_decay}/` (21 CSVs each, 168 rows each, complete).
  - Deterministic DNN Training: `results/shipsnet/deterministic/fold{2..5}/{00..03}/` (complete).
  - Deterministic DNN SEU: Fold 1 is under `results/shipsnet/deterministic/*_SEU/` (complete); Folds 2–5 are scaffolded under `results/shipsnet/seu/deterministic/` (pending).

### 6.2 EuroSAT Layout & The `seu_clean/` Rule
- **BNN Training**: `results/eurosat/bayesian/results_eurosat_v02_{00,01,02,03}/` (252 models complete).
- **SEU Evaluations**:
  - `results/eurosat/seu_clean/v02_{00..03}/` (**CRITICAL**: Always use `seu_clean/` for analysis and aggregation!) (252 CSVs, 168 rows each, complete).
  - `results/eurosat/seu/v02_{00..03}/` (Raw outputs containing historical duplicate appended passes).
- **Why `seu_clean/` is mandatory**:
  Legacy evaluation scripts opened output CSVs in append mode without combo-level resume guards. Re-running configs appended duplicate 168-combo passes. While `set`-based completeness checks masked this, pandas/aggregation scripts concatenating rows will silently double-weight duplicates. `scripts/dedup_seu_csvs.py` generated `seu_clean/` to preserve exactly one complete 168-row pass per configuration.

### 6.3 Master Result Index & Progress Tracking
Consult **`result_index.md`** for the complete factorial grid, CSV file index, schema details, and explicit checklist of finished vs. pending experimental runs (e.g. ShipsNet Folds 2–5 DNN SEU evaluations).


---

## 7. Operational Scripts & Workflows

### 7.1 Training
```bash
# ShipsNet BNN training
uv run python scripts/train_shipsnet.py

# EuroSAT BNN training
uv run python scripts/train_eurosat.py

# Deterministic CNN training
uv run python scripts/train_deterministic.py
```

### 7.2 SEU Robustness Evaluation
```bash
# ShipsNet SEU evaluation
uv run python scripts/eval_seu_shipsnet.py

# EuroSAT SEU evaluation (unseeded standard)
uv run python scripts/eval_seu_eurosat.py

# Deterministic SEU evaluation (Fold 1 single-run or custom)
uv run python scripts/eval_seu_deterministic.py

# Deterministic SEU evaluation (Folds 2 to 5 full sweep)
uv run python scripts/eval_seu_deterministic_folds2to5.py
```

### 7.3 Data Deduplication & Aggregation
```bash
# Deduplicate EuroSAT SEU CSVs into seu_clean/
uv run python scripts/dedup_seu_csvs.py

# Aggregate SEU results across models and folds
uv run python scripts/aggregate_seu.py
uv run python scripts/aggregate_seu_folds.py
```

---

## 8. Important Cautions & Edge Cases

1. **Best-Checkpoint vs. Last-Epoch Evaluation**:
   - Training scripts historically logged test accuracy on the *last epoch* (epoch 100) model state, whereas SEU evaluation scripts load the *best train accuracy* checkpoint (`*_epoch_best_*`).
   - Consult `docs/plan/best_checkpoint_test_accuracy_fix.md` before regenerating final paper accuracy tables.
2. **Opt-in Fast SmartPool (`src/models/components_fast.py`)**:
   - `SmartPoolFast` provides ~19× faster pooling using dual `max_pool2d` operations instead of unfold/topk. It is opt-in via `--fast-smartpool` and bit-identical to standard `SmartPool`. Standard `SmartPool` in `components.py` remains the default.
3. **Opt-in Seeded SEU Script (`scripts/eval_seu_eurosat_seeded.py`)**:
   - Unseeded MC-10 is the historical standard across all EuroSAT `v02_00`–`v02_03` results. Do not switch to `eval_seu_eurosat_seeded.py` for standard aggregations unless exact deterministic regeneration is explicitly requested.
4. **Uniform Prior Loss Divergence**:
   - In uniform prior configurations, `UniformReal` in `src/models/components.py` can produce `inf` ELBO losses when variational guides sample outside the nominal bounds. Refer to `docs/plan/best_checkpoint_test_accuracy_fix.md` when analyzing uniform prior stability.
5. **Backlog & Task Management**:
   - When Backlog MCP is active, consult `backlog://workflow/overview` before initiating large multi-step refactors or project tracking tasks.
