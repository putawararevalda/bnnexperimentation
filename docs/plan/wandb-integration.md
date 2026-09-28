# Plan: WandB Integration (Local Self-Hosted)

## Context

After the codebase refactor, training runs save artifacts (CSVs, PNGs, JSON configs) to
`results/shipsnet/bayesian/` but there is no unified experiment tracking UI. This plan
integrates WandB in **offline-first mode** (no cloud, all data local) with an optional
self-hosted Docker server for a live UI.

The integration touches two files:
- `src/training/svi.py` — the single training loop used by all scripts
- `scripts/train_shipsnet.py` — passes hyperparams into the tracker

---

## Deployment: Two Modes

### Mode A — Offline (no Docker, no account)
Runs are stored as files under `wandb/` at the project root. No live UI.
View runs by syncing to wandb.ai later, or use Mode B.

### Mode B — Self-Hosted Server (Docker, live UI)
Full WandB UI at `http://localhost:8080`. Runs stream live during training.
Requires Docker Desktop to be running.

```
wandb.ai cloud
     ↑ (optional sync)
wandb/ (local run files)    ←  src/training/svi.py logs here
     ↑
Docker container: wandb/local   →  http://localhost:8080 (UI)
```

---

## Step 1 — Install WandB

Add to `requirements.txt`:
```
wandb>=0.17
```

Or install ad-hoc:
```bash
uv run --with wandb python -c "import wandb; print(wandb.__version__)"
```

---

## Step 2 — Configure `.env`

Add these lines to `.env`:

```env
# WandB mode: "offline" (no server) or "online" (Docker server / cloud)
WANDB_MODE=offline

# Local run storage directory
WANDB_DIR=./wandb

# If using Docker self-hosted server (Mode B), uncomment:
# WANDB_MODE=online
# WANDB_BASE_URL=http://localhost:8080
# WANDB_API_KEY=local-<your-key-from-localhost:8080>
```

---

## Step 3 — Instrument `src/training/svi.py`

### 3a. Add import at top of file

```python
import wandb
```

### 3b. Initialize a run at the start of `train_svi_with_stats()`

Insert after `os.makedirs(save_dir, exist_ok=True)`:

```python
aim_run = wandb.init(
    project="bnn-seu-shipsnet",
    name=f"{act_name}_{prior_name}_{timestamp}",
    config={
        "activation": act_name,
        "prior": prior_name,
        "num_epochs": num_epochs,
        "prior_mu": model.prior_mu.item() if hasattr(model, "prior_mu") else None,
        "prior_b": model.prior_b.item() if hasattr(model, "prior_b") else None,
        "batch_size": train_loader.batch_size,
        "train_size": len(train_loader.dataset),
    },
    dir=save_dir,
    reinit=True,
)
```

> `reinit=True` is needed because `train_shipsnet.py` calls `train_svi_with_stats()`
> in a loop across multiple (activation, prior, b) combinations in a single process.

### 3c. Log ELBO loss every epoch

After `epoch_losses.append(avg_loss)`:

```python
wandb.log({"loss/elbo": avg_loss}, step=epoch)
```

### 3d. Log train accuracy at accuracy-check epochs

After `print(f"  Train accuracy: {acc * 100:.2f}%")`:

```python
wandb.log({"accuracy/train": acc}, step=epoch)
```

### 3e. Log best accuracy when a new best is saved

After `print(f"  >> New best ({acc * 100:.2f}%) - saved artifacts")`:

```python
wandb.log({"accuracy/best_train": best_acc}, step=epoch)
```

### 3f. Log test accuracy and close run

In `scripts/train_shipsnet.py`, after `predict_data()` and computing `test_acc`:

```python
wandb.log({"accuracy/test": test_acc})
wandb.finish()
```

---

## Step 4 — (Mode B only) Launch Docker Server

Install [Docker Desktop](https://www.docker.com/products/docker-desktop/) and start it,
then run once:

```bash
docker run --rm -d \
  -v wandb_data:/vol \
  -p 8080:8080 \
  --name wandb-local \
  wandb/local
```

- UI available at `http://localhost:8080`
- On first visit, create a local account and copy the API key
- Add to `.env`:
  ```env
  WANDB_MODE=online
  WANDB_BASE_URL=http://localhost:8080
  WANDB_API_KEY=local-<your-key>
  ```

To stop the server:
```bash
docker stop wandb-local
```

To restart (data persists in the `wandb_data` Docker volume):
```bash
docker start wandb-local
```

---

## Step 5 — Run Training

```bash
uv run --with pyro-ppl,torch,torchvision,tqdm,scikit-learn,python-dotenv,requests,matplotlib,pandas,numpy,wandb \
  python scripts/train_shipsnet.py \
  --prior Gaussian_prior \
  --epoch 100 \
  --b-set full \
  --save-dir results/shipsnet/bayesian
```

Each (activation × b) combination becomes a separate WandB run, all grouped under
the `bnn-seu-shipsnet` project.

---

## Step 6 — (Mode A only) Sync offline runs to server

If using offline mode and want to view in the Docker UI later:

```bash
uv run --with wandb wandb sync wandb/
```

Or sync to wandb.ai cloud (requires account):
```bash
uv run --with wandb wandb sync wandb/ --include-offline
```

---

## What You See in the UI

| View | What it shows |
|------|---------------|
| **Runs table** | All experiments as rows — filter/sort by activation, prior, b, test accuracy |
| **Loss curves** | `loss/elbo` per epoch, overlaid across runs |
| **Accuracy curves** | `accuracy/train` and `accuracy/test` per run |
| **Parallel coordinates** | Best view for comparing prior × activation × b combinations |
| **Scatter plot** | test accuracy vs. prior_b, coloured by activation |

---

## Files Changed

| File | Change |
|------|--------|
| `src/training/svi.py` | Add `wandb.init()`, `wandb.log()` calls |
| `scripts/train_shipsnet.py` | Add `wandb.log(test_acc)` + `wandb.finish()` after `predict_data()` |
| `.env` | Add `WANDB_MODE`, `WANDB_DIR` (and optionally `WANDB_BASE_URL`, `WANDB_API_KEY`) |
| `requirements.txt` | Add `wandb>=0.17` |

No changes needed to `src/models/`, `src/data/`, or any evaluation scripts.

---

## Verification

1. Run training with `--trial-mode --epoch 3`
2. Check `wandb/` directory was created at project root
3. **Mode A**: run `wandb sync wandb/` — confirm run uploads
4. **Mode B**: open `http://localhost:8080` — confirm run appears live during training,
   loss curve updates each epoch, test accuracy logged at end
