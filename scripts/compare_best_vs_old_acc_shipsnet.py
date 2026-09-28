"""
Compare ShipsNet fold-1 "initial test accuracy" across the three places it exists.

Three numbers claim to be the test accuracy of the same run:

1. `old_last_epoch_acc` - recomputed exactly from the existing
   `predictions_*_NN.csv` files in the training dir. These were written by the
   pre-fix `train_shipsnet.py`, so they describe the **last-epoch** model. This
   is what paper Table 1 reports.
2. `recomputed_best_acc` - output of `scripts/eval_best_test_acc_shipsnet.py`,
   i.e. the **best** checkpoint under MC-10 inference.
3. `seu_initial_acc` - the `initial_accuracy` column of the SEU CSVs, which is
   the baseline every SEU delta is measured against. The SEU scripts load the
   **best** checkpoint, so this should agree with (2) up to MC sampling noise.

The two questions this answers:
  - Does (2) reproduce (3)? If yes, the SEU baselines are confirmed consistent
    and the recomputation is trustworthy.
  - How far is (1) from (2)? That is the correction paper Table 1 needs.

Read-only: writes a single comparison CSV to `--out` and prints a summary.

Usage
-----
    uv run --with pandas,numpy python scripts/compare_best_vs_old_acc_shipsnet.py \\
        --recomputed-csv results/shipsnet/best_checkpoint_acc/v02_00/best_checkpoint_test_acc_fold1.csv \\
        --search-dir results/shipsnet/bayesian/results_shipsnet_v02_00 \\
        --seu-dir results/shipsnet/seu/results_shipsnet_v02_00_SEU \\
        --out results/shipsnet/best_checkpoint_acc/v02_00/acc_comparison_fold1.csv
"""
import argparse
import glob
import os
import re

import numpy as np
import pandas as pd

# `predictions_<activation>_<prior>_<YYYYmmdd>_<HHMMSS>_<rounded_acc>.csv`
PREDICTIONS_RE = re.compile(
    r"^predictions_(?P<tag>.+)_(?P<ts>\d{8}_\d{6})_(?P<acc>\d+)\.csv$"
)

JOIN_KEYS = ["activation_fn", "prior", "prior_b"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare old last-epoch vs recomputed best-checkpoint accuracy."
    )
    parser.add_argument(
        "--recomputed-csv", type=str,
        default="results/shipsnet/best_checkpoint_acc/v02_00/best_checkpoint_test_acc_fold1.csv",
    )
    parser.add_argument(
        "--search-dir", type=str,
        default="results/shipsnet/bayesian/results_shipsnet_v02_00",
        help="Training dir holding the original predictions_*.csv files.",
    )
    parser.add_argument(
        "--seu-dir", type=str,
        default="results/shipsnet/seu/results_shipsnet_v02_00_SEU",
        help="SEU results dir, for the initial_accuracy baseline. Optional.",
    )
    parser.add_argument(
        "--out", type=str,
        default="results/shipsnet/best_checkpoint_acc/v02_00/acc_comparison_fold1.csv",
    )
    return parser.parse_args()


def load_old_last_epoch(search_dir: str) -> pd.DataFrame:
    """Recompute exact last-epoch test accuracy from the stored predictions."""
    rows = []
    for path in sorted(glob.glob(os.path.join(search_dir, "predictions_*.csv"))):
        match = PREDICTIONS_RE.match(os.path.basename(path))
        if not match:
            continue
        preds = pd.read_csv(path)
        acc = float((preds["True Label"] == preds["Predicted Label"]).mean())
        rows.append({
            "timestamp": match.group("ts"),
            "old_last_epoch_acc": acc,
            "old_acc_in_filename": int(match.group("acc")) / 100.0,
            "n_test": len(preds),
        })
    if not rows:
        raise FileNotFoundError(f"no predictions_*.csv found in {search_dir}")
    return pd.DataFrame(rows)


def load_seu_initial(seu_dir: str) -> pd.DataFrame:
    """Pull the per-config `initial_accuracy` baseline out of the SEU CSVs."""
    files = sorted(glob.glob(os.path.join(seu_dir, "*.csv")))
    if not files:
        return pd.DataFrame(columns=JOIN_KEYS + ["seu_initial_acc"])

    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    grouped = df.groupby(JOIN_KEYS)["initial_accuracy"].agg(["nunique", "first"])

    inconsistent = grouped[grouped["nunique"] > 1]
    if not inconsistent.empty:
        print(
            f"WARNING: {len(inconsistent)} SEU config(s) carry more than one "
            f"initial_accuracy value; using the first of each:\n{inconsistent}"
        )

    return (
        grouped.reset_index()
        .rename(columns={"first": "seu_initial_acc"})
        .drop(columns=["nunique"])
    )


def summarise(df: pd.DataFrame, new_col: str, old_col: str, label: str,
              n_test: int) -> None:
    """Print a delta summary between two accuracy columns."""
    sub = df.dropna(subset=[new_col, old_col])
    if sub.empty:
        print(f"\n{label}: no overlapping rows to compare.")
        return

    delta = sub[new_col] - sub[old_col]
    abs_delta = delta.abs()
    identical = int((abs_delta < 1e-12).sum())

    print(f"\n{label}  (n={len(sub)})")
    print(f"  identical            : {identical}/{len(sub)}")
    print(f"  mean delta           : {delta.mean() * 100:+.4f} pp")
    print(f"  mean |delta|         : {abs_delta.mean() * 100:.4f} pp")
    print(f"  max  |delta|         : {abs_delta.max() * 100:.4f} pp "
          f"(~{abs_delta.max() * n_test:.1f} test samples)")
    print(f"  {new_col} higher in  : {int((delta > 0).sum())}/{len(sub)}")

    worst = sub.loc[abs_delta.idxmax()]
    print(f"  largest gap          : {worst['activation_fn']}/{worst['prior']}"
          f"/b={worst['prior_b']}  "
          f"{worst[old_col] * 100:.2f}% -> {worst[new_col] * 100:.2f}%")


def main() -> None:
    args = parse_args()

    recomputed = pd.read_csv(args.recomputed_csv)
    old = load_old_last_epoch(args.search_dir)
    seu = load_seu_initial(args.seu_dir) if os.path.isdir(args.seu_dir) else pd.DataFrame()

    df = recomputed.merge(old, on="timestamp", how="left")
    if not seu.empty:
        df = df.merge(seu, on=JOIN_KEYS, how="left")
    else:
        df["seu_initial_acc"] = np.nan
        print(f"NOTE: SEU dir not found ({args.seu_dir}); skipping that comparison.")

    df["delta_best_vs_last"] = df["recomputed_test_acc"] - df["old_last_epoch_acc"]
    df["delta_best_vs_seu"] = df["recomputed_test_acc"] - df["seu_initial_acc"]

    n_test = int(df["n_test"].dropna().iloc[0]) if df["n_test"].notna().any() else 0
    mc_noise = float(df["recomputed_test_acc_std"].max()) if "recomputed_test_acc_std" in df else 0.0

    print(f"Configs evaluated : {len(df)}")
    if n_test:
        print(f"Test set size     : {n_test} samples "
              f"(1 sample = {100 / n_test:.4f} pp)")
    if mc_noise > 0:
        print(f"Max MC noise (std): {mc_noise * 100:.4f} pp across repeats")
    else:
        print("Max MC noise (std): not measured (--repeats 1); "
              "treat small deltas with caution")

    summarise(df, "recomputed_test_acc", "seu_initial_acc",
              "RECOMPUTED (best ckpt) vs SEU initial_accuracy  [expect ~equal]", n_test)
    summarise(df, "recomputed_test_acc", "old_last_epoch_acc",
              "RECOMPUTED (best ckpt) vs OLD last-epoch        [expect a shift]", n_test)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    cols = [
        "timestamp", "activation_fn", "prior", "prior_b", "variant", "fold",
        "best_train_accuracy", "best_accuracy_at_epoch",
        "old_last_epoch_acc", "recomputed_test_acc", "recomputed_test_acc_std",
        "seu_initial_acc", "delta_best_vs_last", "delta_best_vs_seu", "n_test",
    ]
    df[[c for c in cols if c in df.columns]].to_csv(args.out, index=False)
    print(f"\nWrote comparison to {args.out}")


if __name__ == "__main__":
    main()
