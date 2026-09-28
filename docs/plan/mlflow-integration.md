# Plan: MLflow Integration (Local, Result Organization)

## Context

Training runs already save `config_*.json`, `accuracy_*.csv`, and `losses_*.csv` to
`results/shipsnet/bayesian/`. The problem is there is no unified view to compare all
runs — you have to manually open files. MLflow solves this with a local runs table UI,
no Docker, no account, no cloud.

**Core need:** organize and compare results (prior × activation × b) by test accuracy,
not live monitoring.

---

## What MLflow Adds

- `mlflow ui` — one command, local web UI at `http://localhost:5000`
- Runs table with sortable/filterable columns (activation, prior, b, test accuracy)
- Metric plots per run (loss curve, accuracy curve)
- Artifact links to saved model files
- All data stored in `mlruns/` at the project root — no server process needed to log,
  only to view

---

## Step 1 — Install

Add to `requirements.txt`:
```
mlflow>=2.13
```

Or ad-hoc:
```bash
uv run --with mlflow python -c "import mlflow; print(mlflow.__version__)"
```

---

## Step 2 — Instrument `src/training/svi.py`

### 2a. Add import at top of file

```python
import mlflow
```

### 2b. Start a run at the beginning of `train_svi_with_stats()`

Insert after `os.makedirs(save_dir, exist_ok=True)`:

```python
mlflow.set_experiment("bnn-seu-shipsnet")
mlflow.start_run(run_name=f"{act_name}_{prior_name}_{timestamp}")
mlflow.log_params({
    "activation": act_name,
    "prior": prior_name,
    "prior_mu": model.prior_mu.item() if hasattr(model, "prior_mu") else None,
    "prior_b": model.prior_b.item() if hasattr(model, "prior_b") else None,
    "num_epochs": num_epochs,
    "batch_size": train_loader.batch_size,
    "train_size": len(train_loader.dataset),
})
```

### 2c. Log ELBO loss every epoch

After `epoch_losses.append(avg_loss)`:

```python
mlflow.log_metric("loss_elbo", avg_loss, step=epoch)
```

### 2d. Log train accuracy at accuracy-check epochs

After `print(f"  Train accuracy: {acc * 100:.2f}%")`:

```python
mlflow.log_metric("train_acc", acc, step=epoch)
```

### 2e. Log best accuracy when artifacts are saved

After `print(f"  >> New best ({acc * 100:.2f}%) - saved artifacts")`:

```python
mlflow.log_metric("best_train_acc", best_acc, step=epoch)
```

### 2f. End the run at the bottom of `train_svi_with_stats()`

Just before the `return` statement:

```python
mlflow.end_run()
```

---

## Step 3 — Log test accuracy in `scripts/train_shipsnet.py`

After computing `test_acc` from `predict_data()`:

```python
mlflow.log_metric("test_acc", test_acc)
```

> `mlflow.end_run()` in `svi.py` is called after this returns, so the metric is
> captured inside the same run. No extra `start_run` needed in the script.

---

## Step 4 — Launch the UI

```bash
# Run from D:\bnnexperimentation\
uv run --with mlflow mlflow ui
# Opens at http://localhost:5000
```

Keep this terminal open while browsing. Runs already logged appear immediately —
no need to re-run training.

---

## Step 5 — Run Training

```bash
uv run --with pyro-ppl,torch,torchvision,tqdm,scikit-learn,python-dotenv,requests,matplotlib,pandas,numpy,mlflow \
  python scripts/train_shipsnet.py \
  --prior Gaussian_prior \
  --epoch 100 \
  --b-set full \
  --save-dir results/shipsnet/bayesian
```

Each (activation × b) combination logs as a separate run under the
`bnn-seu-shipsnet` experiment.

---

## What You See in the UI

| View | What it shows |
|------|---------------|
| **Runs table** | All experiments as rows — sort by `test_acc` to find best combinations |
| **Columns** | activation, prior, prior_b, num_epochs, best_train_acc, test_acc |
| **Run detail** | loss_elbo and train_acc curves per epoch |
| **Compare** | Select multiple runs → overlay metric plots side by side |

---

## Storage

MLflow stores everything in `mlruns/` at the project root:

```
mlruns/
└── <experiment-id>/
    └── <run-id>/
        ├── params/       ← activation, prior, b, ...
        ├── metrics/      ← loss_elbo, train_acc, test_acc
        └── artifacts/    ← (optional) model files
```

No external process or database needed. The `mlruns/` directory is self-contained —
back it up or move it freely.

Add to `.gitignore`:
```
mlruns/
```

---

## Files Changed

| File | Change |
|------|--------|
| `src/training/svi.py` | Add `mlflow.start_run()`, `log_params()`, `log_metric()`, `end_run()` |
| `scripts/train_shipsnet.py` | Add `mlflow.log_metric("test_acc", ...)` after `predict_data()` |
| `requirements.txt` | Add `mlflow>=2.13` |
| `.gitignore` | Add `mlruns/` |

---

## Verification

1. Run `--trial-mode --epoch 3`
2. Check `mlruns/` directory was created
3. Run `mlflow ui` and open `http://localhost:5000`
4. Confirm run appears with correct params (activation=relu, prior=gaussian, prior_b=1.0)
5. Confirm `loss_elbo` logged for epochs 1–3 and `test_acc` logged at end
