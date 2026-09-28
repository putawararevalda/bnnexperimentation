"""Compare SEU robustness between EuroSAT v00 and ShipsNet fold 1 (variant 0).

Reproduces the tables in `docs/analysis/eurosat_v00_vs_shipsnet_fold1_seu.md`.

Metrics follow the paper:
    AAAD = mean |accuracy_after_seu - initial_accuracy|
    ASD  = mean softmax_difference
    ARIn = sqrt((AAAD^2 + ASD^2) / 2)   -- computed from the *aggregated* AAAD/ASD.

Usage:
    uv run --with pandas,numpy python scripts/compare_seu_datasets.py
"""

from __future__ import annotations

import glob
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

EUROSAT_GLOB = "results/eurosat/seu/v02_00/*.csv"
SHIPSNET_GLOB = "results/shipsnet/seu/results_shipsnet_v02_00_SEU/*.csv"

CENTER_PARAMS = ("locs", "lows")  # gaussian/laplace loc, uniform low
SHIPSNET_PRIOR_B = 1.0  # ShipsNet fold 1 only ran b=1.0; match it for fair comparison


def load_seu(pattern: str, name: str) -> pd.DataFrame:
    """Load and concatenate SEU result CSVs, keeping only valid injections."""
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No SEU CSVs matched: {pattern}")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df["dataset"] = name
    # Rows with a null accuracy_after_seu are skipped injections (negative scale/width).
    df = df[df.accuracy_after_seu.notna()].copy()
    df["aad"] = (df.accuracy_after_seu - df.initial_accuracy).abs()
    df["sd"] = df.softmax_difference
    df["ptype"] = np.where(df.param_type.isin(CENTER_PARAMS), "C", "S")
    # EuroSAT logs activations as '_actWG'; ShipsNet as 'actWG'.
    df["act"] = df.activation_fn.str.lstrip("_")
    return df


def aggregate(df: pd.DataFrame, keys: Optional[List[str]] = None) -> pd.DataFrame:
    """Aggregate AAAD/ASD (and derive ARIn) overall or grouped by `keys`."""
    if keys is None:
        out = pd.DataFrame({"AAAD": [df.aad.mean()], "ASD": [df.sd.mean()], "n": [len(df)]})
    else:
        out = df.groupby(keys).agg(AAAD=("aad", "mean"), ASD=("sd", "mean"), n=("aad", "size"))
    out["ARIn"] = np.sqrt((out.AAAD**2 + out.ASD**2) / 2)
    return out


def print_side_by_side(
    title: str, keys: List[str], pairs: List[Tuple[str, pd.DataFrame]]
) -> None:
    """Print one aggregation for each dataset, side by side."""
    frames = []
    for name, df in pairs:
        a = aggregate(df, keys)
        a.columns = pd.MultiIndex.from_product([[name], a.columns])
        frames.append(a)
    print(f"\n### {title}")
    print(pd.concat(frames, axis=1).round(4).to_string())


def main() -> None:
    pd.set_option("display.width", 220)

    eurosat_all = load_seu(EUROSAT_GLOB, "EuroSAT")
    shipsnet = load_seu(SHIPSNET_GLOB, "ShipsNet")
    eurosat = eurosat_all[np.isclose(eurosat_all.prior_b, SHIPSNET_PRIOR_B)].copy()

    pairs = [("EuroSAT_b1", eurosat), ("ShipsNet_f1", shipsnet)]

    print("### Overall (model variant 0)")
    rows = []
    for name, df in pairs + [("EuroSAT_allb", eurosat_all)]:
        a = aggregate(df)
        a.index = [name]
        a["init_acc"] = df.initial_accuracy.mean()
        rows.append(a)
    print(pd.concat(rows).round(4).to_string())

    print_side_by_side("By prior x param (C=center, S=scale)", ["prior", "ptype"], pairs)
    print_side_by_side("By prior (pooled C+S)", ["prior"], pairs)
    print_side_by_side("By activation", ["act"], pairs)
    print_side_by_side(
        "By layer x parameter", ["location_layer", "location_module"], pairs
    )
    print_side_by_side("By attacked bit index", ["bit_index"], pairs)
    print_side_by_side("By original bit value", ["original_bit_condition"], pairs)

    print("\n### EuroSAT only: effect of prior scale b")
    print(aggregate(eurosat_all, ["prior_b"]).round(4).to_string())

    print("\n### Baseline (initial) accuracy per config")
    for name, df in pairs:
        print(f"\n{name}: mean {df.initial_accuracy.mean():.4f}")
        print(
            df.groupby(["act", "prior"])
            .initial_accuracy.first()
            .unstack()
            .round(4)
            .to_string()
        )


if __name__ == "__main__":
    main()
