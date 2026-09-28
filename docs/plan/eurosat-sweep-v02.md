# EuroSAT BNN Sweep (v02 — All Variants) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run the full BNN training sweep (7 activations × 3 priors × 3 b values = 63 runs) on the EuroSAT 10-class dataset for all 4 model variants (base, smartpool, dropout, weight decay), mirroring the ShipsNet v02_00–03 structure, with MLflow tracking and backfill of existing EuroSAT results.

**Architecture:** Rewrite `scripts/train_eurosat.py` to mirror `scripts/train_shipsnet.py` — add `--variant` flag for the 4 model variants, full 3-prior sweep including Uniform, MLflow integration via `bnn-seu-eurosat` experiment, and variant-specific save directories. A separate backfill script handles the 66 existing EuroSAT result files.

**Tech Stack:** PyTorch, Pyro SVI, `BayesShipsCNN(num_classes=10)`, AutoNormal / AutoLaplace / AutoUniform guides, MLflow, uv

---

## File Map

| File | Action | What changes |
|------|--------|--------------|
| `scripts/train_eurosat.py` | **Rewrite** | Full sweep + variant flags + MLflow + consistent save dirs |
| `scripts/backfill_mlflow_eurosat.py` | **Create** | Retroactively log 66 existing EuroSAT results |
| `results/eurosat/bayesian/results_eurosat_v02_00/` | **Create dir** | Base variant outputs |
| `results/eurosat/bayesian/results_eurosat_v02_01/` | **Create dir** | Smartpool variant outputs |
| `results/eurosat/bayesian/results_eurosat_v02_02/` | **Create dir** | Dropout variant outputs |
| `results/eurosat/bayesian/results_eurosat_v02_03/` | **Create dir** | Weight decay variant outputs |

No changes needed to `src/` — `BayesShipsCNN`, `src/data/eurosat.py`, and `src/training/svi.py` already support everything required.

---

## Task 1: Rewrite `scripts/train_eurosat.py`

**Files:**
- Modify: `scripts/train_eurosat.py` (full rewrite)

The current script runs only one activation (`relu`) with no sweep, no variant support, no MLflow, and an incompatible config format. Replace it entirely to mirror `train_shipsnet.py`.

- [ ] **Step 1: Replace the file contents**

```python
"""
Train Bayesian CNN on EuroSAT dataset (10-class).

Sweeps over prior distributions x activation functions x prior scale (b) values.
Runs one of four model variants via --variant flag.

Usage examples:
    uv run python scripts/train_eurosat.py --variant 00 --epoch 100
    uv run python scripts/train_eurosat.py --variant 01 --epoch 100 --prior Laplace_prior
    uv run python scripts/train_eurosat.py --variant 00 --trial-mode
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import time
import os

import numpy as np
import pyro
import torch
from pyro.infer import SVI, Trace_ELBO
from pyro.infer.autoguide import AutoNormal
from pyro.optim import ClippedAdam
from sklearn.metrics import confusion_matrix

from src.data.eurosat import load_data
from src.models.bayesian_cnn import BayesShipsCNN
from src.training.svi import train_svi_with_stats, plot_training_results_with_stats, predict_data
from src.utils.guide import AutoLaplace, AutoUniform
from src.utils.notify import send_telegram_message

import pandas as pd


VARIANT_CONFIG = {
    "00": {"smartpool": False, "dropout": False, "weight_decay": False, "label": "base"},
    "01": {"smartpool": True,  "dropout": False, "weight_decay": False, "label": "smartpool"},
    "02": {"smartpool": False, "dropout": True,  "weight_decay": False, "label": "dropout"},
    "03": {"smartpool": False, "dropout": False, "weight_decay": True,  "label": "weight_decay"},
}


def parse_args():
    parser = argparse.ArgumentParser(description="Train Bayesian CNN on EuroSAT (full sweep)")
    parser.add_argument("--variant", type=str, default="00", choices=["00", "01", "02", "03"],
                        help="Model variant: 00=base 01=smartpool 02=dropout 03=weight_decay")
    parser.add_argument("--prior", type=str, default="all",
                        choices=["Gaussian_prior", "Laplace_prior", "Uniform_prior", "all"],
                        help="Prior distribution. Default: all (sweep all three)")
    parser.add_argument("--epoch", type=int, default=100,
                        help="Number of training epochs. Default: 100")
    parser.add_argument("--b-set", type=str, default="full", choices=["full", "single"],
                        help="Prior scale sweep: full=[10.0,1.0,0.1], single=[1.0]. Default: full")
    parser.add_argument("--save-dir", type=str, default=None,
                        help="Override save directory (default: results/eurosat/bayesian/results_eurosat_v02_{variant})")
    parser.add_argument("--trial-mode", dest="trial_mode", action="store_true",
                        help="Run only 1 combination for 1 epoch (quick smoke test)")
    parser.set_defaults(trial_mode=False)
    return parser.parse_args()


def build_guide(prior_dist, model, b, device):
    scale = 0.25 * b
    if prior_dist == "gaussian":
        return AutoNormal(model, init_scale=scale).to(device)
    elif prior_dist == "laplace":
        return AutoLaplace(model, init_scale=scale).to(device)
    elif prior_dist == "uniform":
        return AutoUniform(model, init_scale=scale).to(device)
    raise ValueError(f"Unknown prior_dist: {prior_dist}")


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    variant_cfg = VARIANT_CONFIG[args.variant]
    save_dir = args.save_dir or f"results/eurosat/bayesian/results_eurosat_v02_{args.variant}"
    os.makedirs(save_dir, exist_ok=True)

    prior_map = {
        "Gaussian_prior": "gaussian",
        "Laplace_prior": "laplace",
        "Uniform_prior": "uniform",
    }
    if args.prior == "all":
        prior_list = ["gaussian", "laplace", "uniform"]
    else:
        prior_list = [prior_map[args.prior]]

    activation_list = ["relu", "tanh", "sigmoid", "sinusoidal", "relu6", "wg", "rwg"]
    b_list = [10.0, 1.0, 0.1] if args.b_set == "full" else [1.0]

    if args.trial_mode:
        activation_list = activation_list[:1]
        b_list = b_list[:1]
        prior_list = prior_list[:1]
        args.epoch = 1

    combos = [(a, p, b) for p in prior_list for a in activation_list for b in b_list]
    total = len(combos)
    print(f"Variant: {args.variant} ({variant_cfg['label']})")
    print(f"Total combinations: {total}  epochs={args.epoch}  save_dir={save_dir}")

    for exp_num, (act, prior_dist, b) in enumerate(combos, start=1):
        send_telegram_message(
            title=f"EuroSAT v02_{args.variant} {exp_num}/{total}",
            message=f"activation={act}, prior={prior_dist}, b={b}"
        )

        pyro.clear_param_store()
        t_start = time.time()

        model = BayesShipsCNN(
            num_classes=10, device=device,
            activation=act, prior_dist=prior_dist,
            mu=0.0, b=b,
            smartpool_switch=variant_cfg["smartpool"],
            dropout_switch=variant_cfg["dropout"],
        )
        guide = build_guide(prior_dist, model, b, device)

        wd = 1e-4 if variant_cfg["weight_decay"] else 0.0
        optimizer = ClippedAdam({"lr": 1e-3, "weight_decay": wd})
        svi = SVI(model=model, guide=guide, optim=optimizer,
                  loss=Trace_ELBO(num_particles=1))

        model.to(device)
        guide.to(device)

        train_loader, test_loader = load_data(batch_size=54)

        (losses, accuracies, accuracy_epochs,
         loc_stats, scale_stats,
         best_model_path, best_guide_path, best_ps_path,
         ts) = train_svi_with_stats(
            model, guide, svi, train_loader, device,
            num_epochs=args.epoch,
            save_dir=save_dir,
        )

        act_name = model.activation_fn.__name__ if hasattr(model.activation_fn, "__name__") else str(model.activation_fn)
        plot_training_results_with_stats(
            losses, accuracies, accuracy_epochs,
            loc_stats, scale_stats,
            act_name, prior_dist, ts,
            save_dir=save_dir,
        )

        labels, preds = predict_data(model, guide, test_loader, device, num_samples=10)
        cm = confusion_matrix(labels, preds)
        test_acc = np.trace(cm) / np.sum(cm)
        print(f"Test accuracy: {test_acc * 100:.4f}%")

        try:
            import mlflow as _mlflow
            _mlflow.log_metric("test_acc", test_acc)
            _mlflow.end_run()
        except Exception:
            pass

        pd.DataFrame({"True Label": labels, "Predicted Label": preds}).to_csv(
            os.path.join(save_dir, f"predictions_{act_name}_{prior_dist}_{ts}_{test_acc*100:.0f}.csv"),
            index=False,
        )

        elapsed = time.time() - t_start
        send_telegram_message(
            title=f"EuroSAT v02_{args.variant} {exp_num}/{total} Done",
            message=f"activation={act}, prior={prior_dist}, b={b}\n"
                    f"Test accuracy: {test_acc * 100:.2f}%\n"
                    f"Time: {elapsed:.1f}s"
        )


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify `--help` exits cleanly**

```bash
uv run python scripts/train_eurosat.py --help
```

Expected: prints usage, exits 0. No import errors.

- [ ] **Step 3: Commit**

```bash
git add scripts/train_eurosat.py
git commit -m "refactor(eurosat): rewrite train script with full sweep and variant support"
```

---

## Task 2: Set MLflow experiment to `bnn-seu-eurosat`

**Context:** `src/training/svi.py` hardcodes `mlflow.set_experiment("bnn-seu-shipsnet")`. EuroSAT runs should go into a separate experiment so they don't mix in the UI.

**Files:**
- Modify: `scripts/train_eurosat.py` — add MLflow experiment override before the sweep loop

- [ ] **Step 1: Add MLflow experiment setup in `train_eurosat.py` `main()`, just before the combos loop**

```python
    try:
        import mlflow
        mlflow.set_experiment("bnn-seu-eurosat")
    except Exception:
        pass
```

Insert after the `print(f"Total combinations...")` line and before `for exp_num, ...`.

- [ ] **Step 2: Verify the experiment name is correct**

After running `--trial-mode` (Task 3), check:
```bash
uv run --with mlflow python -c "import mlflow; c = mlflow.MlflowClient(); print([e.name for e in c.search_experiments()])"
```

Expected output includes `"bnn-seu-eurosat"`.

- [ ] **Step 3: Commit**

```bash
git add scripts/train_eurosat.py
git commit -m "feat(eurosat): log to bnn-seu-eurosat MLflow experiment"
```

---

## Task 3: Smoke Test with `--trial-mode`

- [ ] **Step 1: Run trial mode (1 epoch, 1 combination)**

```bash
uv run --with pyro-ppl,torch,torchvision,tqdm,scikit-learn,python-dotenv,requests,matplotlib,pandas,numpy,mlflow python scripts/train_eurosat.py --variant 00 --trial-mode
```

Expected:
- Prints `Variant: 00 (base)  Total combinations: 1  epochs=1`
- Runs 1 epoch of SVI on EuroSAT train set
- Prints `Test accuracy: XX.XXXX%`
- Creates files in `results/eurosat/bayesian/results_eurosat_v02_00/`:
  - `config_relu_gaussian_<ts>.json`
  - `accuracy_results_relu_gaussian_<ts>.csv`
  - `losses_relu_gaussian_<ts>.csv`
  - `predictions_relu_gaussian_<ts>_<acc>.csv`

- [ ] **Step 2: Verify MLflow run was created**

```bash
uv run --with mlflow python -c "
import mlflow
c = mlflow.MlflowClient()
exps = c.search_experiments()
for e in exps:
    runs = c.search_runs(e.experiment_id, max_results=3, order_by=['start_time DESC'])
    print(e.name, len(runs), 'runs')
    for r in runs[:1]:
        print(' ', r.info.run_name, r.data.params, r.data.metrics)
"
```

Expected: `bnn-seu-eurosat` experiment appears with 1 run containing `activation=relu`, `prior=gaussian`, and `test_acc`.

---

## Task 4: Backfill Existing EuroSAT Results into MLflow

**Context:** 66 config JSON files already exist under `results/eurosat/bayesian/`. They have a different format from ShipsNet:
- Timestamp uses dashes: `20250712-160433` instead of `20250712_160433`
- Prior stored as `"Gaussian_prior"` instead of `"gaussian"`
- Scale stored as `"sigma"` instead of `prior_params.b`
- `test_accuracy` stored directly in the config (no predictions CSV needed)
- `activation` key is missing in some older configs (default: `relu`)

**Files:**
- Create: `scripts/backfill_mlflow_eurosat.py`

- [ ] **Step 1: Create the backfill script**

```python
"""
Retroactively log existing EuroSAT training results into MLflow.

Usage:
    uv run --with mlflow python scripts/backfill_mlflow_eurosat.py
    uv run --with mlflow python scripts/backfill_mlflow_eurosat.py --dry-run
"""
import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd
import mlflow

RESULTS_ROOT = Path("results/eurosat/bayesian")
EXPERIMENT_NAME = "bnn-seu-eurosat"

PRIOR_NORMALIZE = {
    "Gaussian_prior": "gaussian",
    "Laplace_prior": "laplace",
    "Uniform_prior": "uniform",
    "gaussian": "gaussian",
    "laplace": "laplace",
    "uniform": "uniform",
}

VARIANT_MAP = {
    "results_GP_eurosat": "early",
    "results_GP_eurosat_old": "early",
    "results_GP_eurosat_old3": "early",
    "results_GP_eurosat_TEST": "ablation",
    "results_GP_eurosat_elbo3": "ablation",
    "results_GP_eurosat_newslate": "guide_iteration",
    "results_eurosat": "early",
    "results_eurosat_v02_00": "paper_final",
    "results_eurosat_v02_01": "paper_final",
    "results_eurosat_v02_02": "paper_final",
    "results_eurosat_v02_03": "paper_final",
}

MODEL_VARIANT_MAP = {
    "results_eurosat_v02_00": "base",
    "results_eurosat_v02_01": "smartpool",
    "results_eurosat_v02_02": "dropout",
    "results_eurosat_v02_03": "weight_decay",
}


def extract_timestamp(filename: str) -> str | None:
    # handles both YYYYMMDD_HHMMSS and YYYYMMDD-HHMMSS
    m = re.search(r"(\d{8}[-_]\d{6})", filename)
    return m.group(1) if m else None


def find_csv(directory: Path, prefix: str, timestamp: str) -> Path | None:
    ts_variants = [timestamp, timestamp.replace("-", "_"), timestamp.replace("_", "-")]
    for ts in ts_variants:
        for f in directory.glob(f"{prefix}*{ts}*.csv"):
            return f
    return None


def load_csv_metrics(csv_path: Path, metric_col: str, epoch_col: str = "epoch") -> list[tuple[int, float]]:
    try:
        df = pd.read_csv(csv_path)
        if epoch_col not in df.columns or metric_col not in df.columns:
            return []
        return list(zip(df[epoch_col].astype(int), df[metric_col].astype(float)))
    except Exception:
        return []


def backfill_run(config_path: Path, dry_run: bool) -> bool:
    directory = config_path.parent
    dir_name = directory.name

    with open(config_path, encoding="utf-8") as f:
        config = json.load(f)

    timestamp = extract_timestamp(config_path.name)
    if timestamp is None:
        print(f"  SKIP  {config_path.name} — no timestamp in filename")
        return False

    # Normalize fields — handle both old and new config formats
    act = config.get("activation", "relu")
    prior_raw = config.get("prior", config.get("prior_dist", "gaussian"))
    prior = PRIOR_NORMALIZE.get(prior_raw, prior_raw)
    b = config.get("prior_params", {}).get("b") or config.get("sigma") or config.get("b")
    num_epochs = config.get("num_epochs")
    batch_size = config.get("batch_size", 54)
    train_size = config.get("train_size")
    best_acc = config.get("best_accuracy")
    test_acc = config.get("test_accuracy")

    ts_norm = timestamp.replace("-", "_")
    acc_csv = find_csv(directory, "accuracy_results_", timestamp)
    loss_csv = find_csv(directory, "losses_", timestamp)

    run_name = f"backfill_{act}_{prior}_{ts_norm}"
    run_set = VARIANT_MAP.get(dir_name, "unknown")
    model_variant = MODEL_VARIANT_MAP.get(dir_name, "unknown")

    if dry_run:
        print(
            f"  [dry-run] {run_name}  b={b}  "
            f"best_train={best_acc}  test={test_acc}  "
            f"acc_csv={'yes' if acc_csv else 'no'}  loss_csv={'yes' if loss_csv else 'no'}"
        )
        return True

    with mlflow.start_run(run_name=run_name):
        mlflow.set_tags({
            "source": "backfill",
            "source_dir": dir_name,
            "run_set": run_set,
            "model_variant": model_variant,
        })
        mlflow.log_params({
            "activation": act,
            "prior": prior,
            "prior_b": b,
            "num_epochs": num_epochs,
            "batch_size": batch_size,
            "train_size": train_size,
            "dataset": "eurosat",
        })
        if best_acc is not None:
            mlflow.log_metric("best_train_acc", best_acc)
        if test_acc is not None:
            mlflow.log_metric("test_acc", test_acc)

        if loss_csv:
            for epoch, loss in load_csv_metrics(loss_csv, "loss"):
                mlflow.log_metric("loss_elbo", loss, step=epoch)
        if acc_csv:
            for epoch, acc in load_csv_metrics(acc_csv, "accuracy"):
                mlflow.log_metric("train_acc", acc, step=epoch)

    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--results-root", default=str(RESULTS_ROOT))
    args = parser.parse_args()

    root = Path(args.results_root)
    if not root.exists():
        print(f"ERROR: {root} does not exist")
        sys.exit(1)

    configs = sorted(root.rglob("config_*.json"))
    print(f"Found {len(configs)} config files under {root}")

    if not args.dry_run:
        mlflow.set_experiment(EXPERIMENT_NAME)

    ok = skipped = 0
    for cfg in configs:
        print(f"Processing {cfg.relative_to(root)} ...")
        try:
            if backfill_run(cfg, args.dry_run):
                ok += 1
            else:
                skipped += 1
        except Exception as e:
            print(f"  ERROR: {e}")
            skipped += 1

    print(f"\nDone. Logged={ok}  Skipped={skipped}")
    if not args.dry_run:
        print("Launch UI:  uv run --with mlflow mlflow ui")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Dry-run to verify discovery**

```bash
uv run --with mlflow python scripts/backfill_mlflow_eurosat.py --dry-run
```

Expected: prints each config with `[dry-run]` prefix, `test=<value>` from the config, ends with `Logged=66  Skipped=0` (or similar, depending on actual file count).

- [ ] **Step 3: Run the real backfill**

```bash
uv run --with mlflow python scripts/backfill_mlflow_eurosat.py
```

Expected: ends with `Logged=66  Skipped=0` and `Launch UI: uv run --with mlflow mlflow ui`.

- [ ] **Step 4: Commit**

```bash
git add scripts/backfill_mlflow_eurosat.py
git commit -m "feat(eurosat): add MLflow backfill script for existing results"
```

---

## Task 5: Run the Full Sweep — All 4 Variants

Run each variant in a separate terminal. Each takes ~7–14 hours on CPU (63 combos × 100 epochs). Use GPU if available.

- [ ] **Variant 00 — Base (run first, establishes baseline)**

```bash
uv run --with pyro-ppl,torch,torchvision,tqdm,scikit-learn,python-dotenv,requests,matplotlib,pandas,numpy,mlflow python scripts/train_eurosat.py --variant 00 --epoch 100
```

Saves to: `results/eurosat/bayesian/results_eurosat_v02_00/`

- [ ] **Variant 01 — SmartPool**

```bash
uv run --with pyro-ppl,torch,torchvision,tqdm,scikit-learn,python-dotenv,requests,matplotlib,pandas,numpy,mlflow python scripts/train_eurosat.py --variant 01 --epoch 100
```

Saves to: `results/eurosat/bayesian/results_eurosat_v02_01/`

- [ ] **Variant 02 — Dropout**

```bash
uv run --with pyro-ppl,torch,torchvision,tqdm,scikit-learn,python-dotenv,requests,matplotlib,pandas,numpy,mlflow python scripts/train_eurosat.py --variant 02 --epoch 100
```

Saves to: `results/eurosat/bayesian/results_eurosat_v02_02/`

- [ ] **Variant 03 — Weight Decay**

```bash
uv run --with pyro-ppl,torch,torchvision,tqdm,scikit-learn,python-dotenv,requests,matplotlib,pandas,numpy,mlflow python scripts/train_eurosat.py --variant 03 --epoch 100
```

Saves to: `results/eurosat/bayesian/results_eurosat_v02_03/`

---

## Verification Checklist

After all 4 variants complete:

- [ ] Each `results_eurosat_v02_0X/` has 63 `config_*.json` files
- [ ] MLflow `bnn-seu-eurosat` experiment has 252 runs (4 × 63)
- [ ] Runs have `test_acc`, `loss_elbo`, and `train_acc` metrics
- [ ] Sort by `test_acc` in UI — top results should be ~70–80% (EuroSAT is harder than ShipsNet)
- [ ] Filter `tags.model_variant = "smartpool"` — compare against `"base"` to see SmartPool effect

---

## Key Differences from ShipsNet

| | ShipsNet | EuroSAT |
|--|---------|---------|
| Classes | 2 | 10 |
| Batch size | 16 | 54 |
| Expected test_acc | ~85–95% | ~65–80% |
| MLflow experiment | `bnn-seu-shipsnet` | `bnn-seu-eurosat` |
| Save dir | `results/shipsnet/bayesian/results_shipsnet_v02_0X/` | `results/eurosat/bayesian/results_eurosat_v02_0X/` |
