# Experimental Results Index & Combinations Guide

This document indexes all training artifacts, Single Event Upset (SEU) injection evaluations, and summary tables across the entire experimental grid: **DNN vs. BNN**, **ShipsNet (Folds 1 to 5)**, and **EuroSAT (10-Class)**. Use this guide to track progress and identify next steps.

---

## 1. Experimental Grid Dimensions & Combinations

### 1.1 Factorial Grid

| Dimension | DNN (Deterministic) Baseline | BNN (Bayesian SVI) |
|---|---|---|
| **Architecture Variants** | **4**: `00` (Base), `01` (SmartPool), `02` (Dropout $p=0.5$), `03` (Weight Decay $\lambda=10^{-4}$) | **4**: `00` (Base), `01` (SmartPool), `02` (Dropout $p=0.5$), `03` (Weight Decay $\lambda=10^{-4}$) |
| **Activation Functions** | **7**: `relu`, `tanh`, `sigmoid`, `sin`, `relu6`, `actWG`, `actRWG` | **7**: `relu`, `tanh`, `sigmoid`, `sinusoidal`, `relu6`, `wg`, `rwg` |
| **Prior Distributions** | *None* | **3**: `Gaussian`, `Laplace`, `Uniform` |
| **Prior Scales ($b$)** | *None* | **3**: $\{10.0, 1.0, 0.1\}$ (*Full sweep*) or **1**: $\{1.0\}$ (*Canonical k-fold*) |
| **Models per Split/Fold** | $4 \times 7 =$ **28 models** | **252 models** (*Full sweep*: $4 \times 7 \times 3 \times 3$)<br>**84 models** (*Canonical $b=1.0$*: $4 \times 7 \times 3 \times 1$) |

### 1.2 SEU Injection Grid per Model

Bitflips follow IEEE-754 FP32 across:
- **Locations**: 2 (`beginning` [index 0], `end` [index -1])
- **Layers**: 3 (`conv1`, `conv2`, `fc1`)
- **Modules**: 2 (`weight`, `bias`)
- **Bit positions**: 7 (`0, 1, 3, 6, 10, 15, 21`)
- **Targets**:
  - **DNN**: Direct parameter tensor $\rightarrow 2 \times 3 \times 2 \times 7 =$ **84 injections / model**
  - **BNN**: 2 variational parameters (`locs` & `scales`, or `lows` & `widths`) $\rightarrow 2 \times 3 \times 2 \times 7 \times 2 =$ **168 injections / model**

### 1.3 Total Combinations Summary

| Experiment Partition | DNN Trainings | BNN Trainings | DNN SEU Injections | BNN SEU Injections |
|---|---|---|---|---|
| **EuroSAT (1 split)** | 28 *(planned)* | **252** *(complete)* | 2,352 *(planned)* | **42,336** *(complete)* |
| **ShipsNet Fold 1 (Paper Baseline)** | **28** *(complete)* | **252** *(complete)* | **2,352** *(complete)* | **42,336** *(complete)* |
| **ShipsNet Folds 1–5 (Canonical $b=1.0$)** | **140** *(trained)* | **420** *(complete)* | 11,760 *(Fold 1 done; Folds 2–5 pending)* | **70,560** *(complete)* |

---

## 2. Directory Layout & CSV File Index

### 2.1 EuroSAT (10-Class Single Stratified Split)

#### BNN (Bayesian Neural Networks)
- **Training Artifacts & Logs**: `results/eurosat/bayesian/results_eurosat_v02_{00..03}/`
  - `accuracy_results_{activation}_{prior}_{timestamp}.csv` (Test accuracy per 10 epochs)
  - `losses_{activation}_{prior}_{timestamp}.csv` (ELBO loss per epoch)
  - `predictions_{activation}_{prior}_{timestamp}.csv` (MC test predictions)
  - `config_{activation}_{prior}_{timestamp}.json`
  - Model weights: `model_best_*`, `model_final_*`, `param_store_best_*`, `param_store_final_*`
  - *Total: 63 models per variant $\times$ 4 variants = 252 models.*
- **SEU Evaluations**:
  - **Canonical Deduplicated Path (ALWAYS USE THIS)**: `results/eurosat/seu_clean/v02_{00..03}/_{timestamp}.csv`
    - 63 CSV files per variant folder $\times$ 4 variants = **252 CSV files**.
    - Each file contains exactly **168 SEU injection rows**.
  - *(Raw/duplicate runs reside in `results/eurosat/seu/v02_{00..03}/` - do not use for aggregation).*

#### DNN (Deterministic Baselines)
- **Training Directory**: `results/eurosat/deterministic/`
  - Contains preliminary validation checkpoints (e.g., `det_relu_30ep_v2split.pth`).
  - Full 28-model grid and SEU evaluation are currently pending.

---

### 2.2 ShipsNet: Fold 1 Baseline (Original Paper Run)

Stored directly at the root results folders (not under a `fold1/` directory):

#### BNN
- **Training Artifacts & Logs**: `results/shipsnet/bayesian/results_shipsnet_v02_{00..03}/`
  - 63 models per variant $\times$ 4 variants = **252 models** (covering all 3 scales $b \in \{10.0, 1.0, 0.1\}$).
  - Includes `accuracy_results_*.csv`, `losses_*.csv`, `predictions_*.csv`, and `config_*.json`.
- **SEU Evaluations**: `results/shipsnet/seu/results_shipsnet_v02_{00..03}_SEU/_{timestamp}.csv`
  - 21 CSV files per variant $\times$ 4 variants (focused on $b=1.0$ baseline).
  - Each file has **168 SEU injection rows**.

#### DNN
- **Training Logs**: `results/shipsnet/deterministic/results_shipsnet_deterministic_{00..03}/`
  - `training_log_{activation}_{timestamp}.csv` (7 files per variant $\times$ 4 variants = **28 files**).
  - Checkpoints: `best_model_{activation}_{timestamp}.pth`.
- **SEU Evaluations**: `results/shipsnet/deterministic/results_shipsnet_deterministic_{00..03}_SEU/_{timestamp}.csv`
  - 7 CSV files per variant $\times$ 4 variants = **28 CSV files**.
  - Each file contains **84 SEU injection rows**.

---

### 2.3 ShipsNet: Folds 1 to 5 Cross-Validation

#### BNN
- **Training Artifacts & Logs**: `results/shipsnet/bayesian/fold{1..5}/{base,dropout,smartpool,weight_decay}/`
  - 21 models per variant $\times$ 4 variants $\times$ 5 folds = **420 models** ($b=1.0$ canonical grid).
  - Each contains `accuracy_results_*.csv`, `losses_*.csv`, `predictions_*.csv`, and `config_*.json`.
- **SEU Evaluations**: `results/shipsnet/seu/fold{1..5}/{base,dropout,smartpool,weight_decay}/_{timestamp}.csv`
  - 21 CSV files per variant $\times$ 4 variants $\times$ 5 folds = **420 CSV files**.
  - Each file contains **168 SEU injection rows**.

#### DNN
- **Training Logs**: `results/shipsnet/deterministic/fold{2..5}/{00,01,02,03}/`
  - `log_{activation}_{timestamp}.csv` (7 activations $\times$ 4 variants $\times$ 4 folds = **112 models trained**).
  - Checkpoints: `best_{activation}_{timestamp}.pth`.
- **SEU Evaluations**:
  - Fold 1: `results/shipsnet/deterministic/results_shipsnet_deterministic_{00..03}_SEU/` (**28 CSVs, complete**).
  - Folds 2–5: Scaffolded under `results/shipsnet/seu/deterministic/fold{2..5}/{00..03}/` (**pending execution**).

---

### 2.4 Aggregated Summary Tables

Statistical fold-reduced tables generated by `scripts/aggregate_seu_folds.py` reside in `results/tables/shipsnet_folds/`:

| File | Content |
|---|---|
| `foldci_overall.csv` / `.tex` | Overall fold-reduced AAD, SoftmaxDiff, and ARIn with 95% CIs ($n=5$). |
| `foldci_activation.csv` / `.tex` | Robustness breakdown across the 7 activation functions. |
| `foldci_prior.csv` / `.tex` | Breakdown across Gaussian, Laplace, and Uniform priors. |
| `foldci_variant.csv` / `.tex` | Breakdown across Base, SmartPool, Dropout, and Weight Decay. |
| `foldci_layer_bit.csv` / `.tex` | Layer $\times$ bit position sensitivity matrix. |
| `foldci_contrasts.csv` / `.tex` | Paired per-fold hypothesis test differences (e.g. WG vs. ReLU). |

---

## 3. CSV File Row Schemas

### BNN SEU Output (`_{timestamp}.csv`, 168 rows)
```csv
activation_fn,prior,variant,best_accuracy,prior_mu,prior_b,param_type,location_index,location_layer,location_module,bit_index,initial_accuracy,accuracy_after_seu,accuracy_change,softmax_difference,mean_abs_difference,original_bit_condition,remarks
```

### DNN SEU Output (`_{timestamp}.csv`, 84 rows)
```csv
fold,activation_fn,model_variant,location_index,location_layer,location_module,bit_index,initial_accuracy,accuracy_after_seu,accuracy_change,softmax_difference,mean_abs_difference,original_bit_condition,remarks
```

---

## 4. Where to Pick Up Progress Next

| Task / Item | Status | Action Needed |
|---|---|---|
| **ShipsNet Folds 1–5 BNN SEU** | **Complete** (420/420 CSVs) | Ready for final paper reporting. |
| **EuroSAT BNN SEU** | **Complete** (252/252 CSVs in `seu_clean/`) | Ready for final paper reporting. |
| **ShipsNet Fold 1 DNN SEU** | **Complete** (28/28 CSVs) | Already evaluated with single-softmax fix. |
| **ShipsNet Folds 2–5 DNN SEU** | **Ready to run** (Script drafted; 112 models indexed) | Run `uv run python scripts/eval_seu_deterministic_folds2to5.py` (or `scripts\eval_seu_deterministic_folds2to5.bat`). Automatically evaluates all 112 checkpoints with resume support, single-softmax calculation, and saves 84-row CSVs to `results/shipsnet/seu/deterministic/fold{2..5}/`. |
| **EuroSAT DNN Baseline** | **Pending** (Only preliminary models) | (Optional) If paper revision requires full EuroSAT DNN comparison, train 28 models and run SEU injector. |
| **Paper Table & Figure Generation** | **Ready / Blocked on Folds 2–5 DNN SEU** | Re-run `scripts/aggregate_seu_folds.py` with `--contrast` once Folds 2–5 DNN SEUs complete to export final LaTeX tables. |
