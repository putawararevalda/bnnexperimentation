"""
Retroactively log all existing ShipsNet training results into MLflow.

Reads every config_*.json under results/shipsnet/bayesian/, matches the
corresponding accuracy and loss CSVs by timestamp, and creates a completed
MLflow run for each.  Predictions CSVs (if present) supply test_acc.

Usage:
    uv run --with mlflow python scripts/backfill_mlflow.py
    uv run --with mlflow python scripts/backfill_mlflow.py --dry-run
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

import pandas as pd
import mlflow

RESULTS_ROOT = Path("results/shipsnet/bayesian")
EXPERIMENT_NAME = "bnn-seu-shipsnet"

# Map directory name fragments to model variant tags
VARIANT_MAP = {
    "v02_00": "base",
    "v02_01": "smartpool",
    "v02_02": "dropout",
    "v02_03": "weight_decay",
    "_00_mvrt": "base_mvrt",
    "_01": "smartpool",
    "_02": "dropout",
    "_03": "weight_decay",
    "scale": "scale",
    "uniform_test": "uniform_test",
    "newslate_guide": "newslate_guide",
    "newslate": "newslate",
    "elbo3": "elbo3",
}


def infer_variant(dir_name: str) -> str:
    for key, label in VARIANT_MAP.items():
        if key in dir_name:
            return label
    return "unknown"


def extract_timestamp(filename: str) -> str | None:
    m = re.search(r"(\d{8}_\d{6})", filename)
    return m.group(1) if m else None


def find_csv(directory: Path, prefix: str, timestamp: str) -> Path | None:
    for f in directory.glob(f"{prefix}*{timestamp}*.csv"):
        return f
    return None


def find_predictions_csv(directory: Path, timestamp: str) -> tuple[Path | None, float | None]:
    for f in directory.glob(f"predictions_*{timestamp}*.csv"):
        parts = f.stem.split("_")
        try:
            test_acc_pct = float(parts[-1])
            return f, test_acc_pct / 100.0
        except (ValueError, IndexError):
            return f, None
    return None, None


def load_csv_metrics(csv_path: Path, metric_col: str, epoch_col: str = "epoch") -> list[tuple[int, float]]:
    try:
        df = pd.read_csv(csv_path)
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

    act = config.get("activation", "unknown")
    prior = config.get("prior", "unknown")
    prior_params = config.get("prior_params", {})

    acc_csv = find_csv(directory, "accuracy_results_", timestamp)
    loss_csv = find_csv(directory, "losses_", timestamp)
    pred_path, test_acc = find_predictions_csv(directory, timestamp)

    run_name = f"backfill_{act}_{prior}_{timestamp}"
    variant = infer_variant(dir_name)

    if dry_run:
        print(
            f"  [dry-run] {run_name}  "
            f"prior_b={prior_params.get('b')}  "
            f"best_train={config.get('best_accuracy', '?'):.4f}  "
            f"test_acc={test_acc!r}  "
            f"acc_csv={'yes' if acc_csv else 'no'}  "
            f"loss_csv={'yes' if loss_csv else 'no'}"
        )
        return True

    with mlflow.start_run(run_name=run_name):
        mlflow.set_tags({
            "source": "backfill",
            "source_dir": dir_name,
            "model_variant": variant,
            "config_file": config_path.name,
        })

        mlflow.log_params({
            "activation": act,
            "prior": prior,
            "prior_mu": prior_params.get("mu"),
            "prior_b": prior_params.get("b"),
            "num_epochs": config.get("num_epochs"),
            "batch_size": config.get("batch_size"),
            "train_size": config.get("train_size"),
        })

        mlflow.log_metric("best_train_acc", config.get("best_accuracy", float("nan")))

        if loss_csv is not None:
            for epoch, loss in load_csv_metrics(loss_csv, "loss"):
                mlflow.log_metric("loss_elbo", loss, step=epoch)

        if acc_csv is not None:
            for epoch, acc in load_csv_metrics(acc_csv, "accuracy"):
                mlflow.log_metric("train_acc", acc, step=epoch)

        if test_acc is not None:
            mlflow.log_metric("test_acc", test_acc)

    return True


def main():
    parser = argparse.ArgumentParser(description="Backfill existing results into MLflow")
    parser.add_argument("--dry-run", action="store_true", help="Print what would be logged without writing")
    parser.add_argument("--results-root", default=str(RESULTS_ROOT), help="Root directory to scan")
    args = parser.parse_args()

    results_root = Path(args.results_root)
    if not results_root.exists():
        print(f"ERROR: {results_root} does not exist. Run from the project root.")
        sys.exit(1)

    config_files = sorted(results_root.rglob("config_*.json"))
    print(f"Found {len(config_files)} config files under {results_root}")

    if not args.dry_run:
        mlflow.set_experiment(EXPERIMENT_NAME)

    ok = skipped = 0
    for cfg in config_files:
        rel = cfg.relative_to(results_root)
        print(f"Processing {rel} ...")
        try:
            success = backfill_run(cfg, dry_run=args.dry_run)
            if success:
                ok += 1
            else:
                skipped += 1
        except Exception as e:
            print(f"  ERROR: {e}")
            skipped += 1

    print(f"\nDone. Logged={ok}  Skipped={skipped}")
    if not args.dry_run:
        print(f"Launch UI with:  uv run --with mlflow mlflow ui")
        print(f"Then open:       http://localhost:5000")


if __name__ == "__main__":
    main()
