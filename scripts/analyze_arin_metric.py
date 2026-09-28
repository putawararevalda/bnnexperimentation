"""
Evaluate ARIn as a composite robustness metric.

Motivation
----------
ARIn is defined as the RMS of AAD and Softmax Difference:

    ARIn = sqrt((AAD^2 + SD^2) / 2)

Reviewer #1 asked why the metric is needed and why it takes this form. The
objection is that AAD and SD live on very different empirical scales, so an
unnormalised quadratic mean is dominated by the larger term and the composite
carries little information beyond it.

This script tests that objection quantitatively and compares ARIn against
candidate alternatives on the groupings the paper actually reports.

Diagnostics
-----------
1. Scale dominance   -- mean(ARIn) vs mean(SD)/sqrt(2); share of ARIn^2 from each term.
2. Redundancy        -- Spearman/Pearson correlation of ARIn with SD and with AAD.
                        If ARIn ~ SD, the composite adds no decision-relevant signal.
3. Rank agreement    -- Kendall tau between the ARIn ranking of each reported grouping
                        and the ranking under SD alone / AAD alone / normalised variants.
                        tau = 1.0 against SD means every conclusion drawn from ARIn
                        could have been drawn from SD alone.
4. Decision flips    -- does the ARIn "winner" of each grouping change under
                        z-score / min-max normalised aggregation?

Candidate aggregators compared
------------------------------
  arin        current: sqrt((AAD^2 + SD^2)/2)      -- unnormalised RMS
  arin_z      RMS of z-scored AAD and SD           -- TOPSIS-consistent
  arin_mm     RMS of min-max scaled AAD and SD     -- TOPSIS-consistent, bounded
  arin_rank   mean of the two within-grouping ranks -- scale-free, ordinal
  sd_only     SD alone                              -- the null hypothesis
  aad_only    AAD alone

Normalisation is fitted WITHIN each grouping, matching how the paper uses the
metric (to rank alternatives inside a table), not across the whole corpus.

Usage
-----
    uv run --with pandas,numpy,scipy python scripts/analyze_arin_metric.py
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd
from scipy import stats

# Slices mirror analyze_seu_noise_floor.py so the two analyses are comparable.
SLICES: list[tuple[str, str, float | None]] = [
    ("ShipsNet fold 1 v02_00", "results/shipsnet/seu/results_shipsnet_v02_00_SEU", None),
    ("ShipsNet fold 1 v02_01", "results/shipsnet/seu/results_shipsnet_v02_01_SEU", None),
    ("ShipsNet fold 1 v02_02", "results/shipsnet/seu/results_shipsnet_v02_02_SEU", None),
    ("ShipsNet fold 1 v02_03", "results/shipsnet/seu/results_shipsnet_v02_03_SEU", None),
    ("EuroSAT v02_00 base", "results/eurosat/seu_clean/v02_00", 1.0),
    ("EuroSAT v02_01 smartpool", "results/eurosat/seu_clean/v02_01", 1.0),
    ("EuroSAT v02_02 dropout", "results/eurosat/seu_clean/v02_02", 1.0),
    ("EuroSAT v02_03 weight_decay", "results/eurosat/seu_clean/v02_03", 1.0),
]

# Groupings the paper reports tables for.
GROUPINGS = ["activation_fn", "prior", "location_layer", "bit_index", "param_type"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ARIn composite-metric evaluation")
    parser.add_argument("--out-dir", type=str, default=None,
                        help="Optional directory for per-diagnostic CSV output.")
    return parser.parse_args()


def load_slice(path: str, prior_b: float | None) -> pd.DataFrame:
    """Load every SEU result CSV under `path` into one frame."""
    files = sorted(glob.glob(os.path.join(path, "*.csv")))
    if not files:
        return pd.DataFrame()
    frames = [pd.read_csv(f) for f in files]
    df = pd.concat(frames, ignore_index=True)
    if prior_b is not None and "prior_b" in df.columns:
        df = df[df["prior_b"] == prior_b]
    df = df.dropna(subset=["accuracy_change", "softmax_difference"])
    # Drop non-finite injections (invalid spread parameter after a sign flip).
    df = df[np.isfinite(df["accuracy_change"]) & np.isfinite(df["softmax_difference"])]
    df["aad"] = df["accuracy_change"].abs()
    df["sd"] = df["softmax_difference"]
    df["arin"] = np.sqrt((df["aad"] ** 2 + df["sd"] ** 2) / 2)
    return df


def scale_dominance(df: pd.DataFrame) -> dict:
    """How much of ARIn is carried by each term."""
    aad_sq = (df["aad"] ** 2).mean()
    sd_sq = (df["sd"] ** 2).mean()
    return {
        "mean_aad": df["aad"].mean(),
        "mean_sd": df["sd"].mean(),
        "mean_arin": df["arin"].mean(),
        "sd_over_sqrt2": df["sd"].mean() / np.sqrt(2),
        "aad_share_of_sq": aad_sq / (aad_sq + sd_sq),
        "sd_share_of_sq": sd_sq / (aad_sq + sd_sq),
        "ratio_arin_to_sd_sqrt2": df["arin"].mean() / (df["sd"].mean() / np.sqrt(2)),
    }


def redundancy(df: pd.DataFrame) -> dict:
    """Correlation of the composite with each of its own components."""
    return {
        "pearson_arin_sd": stats.pearsonr(df["arin"], df["sd"])[0],
        "pearson_arin_aad": stats.pearsonr(df["arin"], df["aad"])[0],
        "spearman_arin_sd": stats.spearmanr(df["arin"], df["sd"])[0],
        "spearman_arin_aad": stats.spearmanr(df["arin"], df["aad"])[0],
        "spearman_aad_sd": stats.spearmanr(df["aad"], df["sd"])[0],
        "r2_arin_from_sd": stats.pearsonr(df["arin"], df["sd"])[0] ** 2,
    }


def aggregators(g: pd.DataFrame) -> pd.DataFrame:
    """Compute every candidate composite on a grouped (already averaged) frame."""
    out = g.copy()

    # Normalisers must preserve direction (lower = better) and keep the ideal
    # point at 0. Centering (z-score) breaks both: squaring a centred value
    # measures distance from the MEAN, so an excellent level with a large
    # negative z is scored as badly as a terrible one. All three below are
    # scale-only or floor-anchored.

    def _vec(s: pd.Series) -> pd.Series:
        """TOPSIS vector normalisation: x_i / sqrt(sum x_j^2)."""
        nrm = np.sqrt((s ** 2).sum())
        return s / nrm if nrm > 0 else s * 0.0

    def _std(s: pd.Series) -> pd.Series:
        """Scale-only standardisation: x / sigma, no centering."""
        sd = s.std(ddof=0)
        return s / sd if sd > 0 else s * 0.0

    def _mm(s: pd.Series) -> pd.Series:
        rng = s.max() - s.min()
        return (s - s.min()) / rng if rng > 0 else s * 0.0

    out["arin"] = np.sqrt((out["aad"] ** 2 + out["sd"] ** 2) / 2)
    out["arin_vec"] = np.sqrt((_vec(out["aad"]) ** 2 + _vec(out["sd"]) ** 2) / 2)
    out["arin_std"] = np.sqrt((_std(out["aad"]) ** 2 + _std(out["sd"]) ** 2) / 2)
    out["arin_mm"] = np.sqrt((_mm(out["aad"]) ** 2 + _mm(out["sd"]) ** 2) / 2)
    out["arin_rank"] = (out["aad"].rank() + out["sd"].rank()) / 2
    out["sd_only"] = out["sd"]
    out["aad_only"] = out["aad"]
    return out


def rank_agreement(df: pd.DataFrame, by: str) -> dict | None:
    """Kendall tau between the ARIn ranking of a grouping and each alternative."""
    if by not in df.columns:
        return None
    g = df.groupby(by)[["aad", "sd"]].mean().reset_index()
    if len(g) < 3:
        return None
    g = aggregators(g)

    res = {"grouping": by, "n_levels": len(g)}
    base = g["arin"].rank()
    for alt in ["sd_only", "aad_only", "arin_vec", "arin_std", "arin_mm", "arin_rank"]:
        tau = stats.kendalltau(base, g[alt].rank())[0]
        res[f"tau_vs_{alt}"] = tau
    # Winner (most robust = lowest) under each aggregator.
    for col in ["arin", "sd_only", "aad_only", "arin_vec", "arin_std", "arin_mm", "arin_rank"]:
        res[f"best_{col}"] = str(g.loc[g[col].idxmin(), by])
    return res


def main() -> None:
    args = parse_args()
    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)

    dom_rows, red_rows, rank_rows = [], [], []
    pooled = []

    for name, path, prior_b in SLICES:
        if not os.path.isdir(path):
            print(f"[skip] {name}: {path} not found")
            continue
        df = load_slice(path, prior_b)
        if df.empty:
            print(f"[skip] {name}: no rows")
            continue
        pooled.append(df.assign(slice=name))

        dom_rows.append({"slice": name, "n": len(df), **scale_dominance(df)})
        red_rows.append({"slice": name, "n": len(df), **redundancy(df)})
        for by in GROUPINGS:
            r = rank_agreement(df, by)
            if r:
                rank_rows.append({"slice": name, **r})

    dom = pd.DataFrame(dom_rows)
    red = pd.DataFrame(red_rows)
    rank = pd.DataFrame(rank_rows)

    pd.set_option("display.width", 200, "display.max_columns", 50)

    print("\n" + "=" * 78)
    print("1. SCALE DOMINANCE  --  is ARIn just SD/sqrt(2)?")
    print("=" * 78)
    print(dom[["slice", "n", "mean_aad", "mean_sd", "mean_arin", "sd_over_sqrt2",
               "ratio_arin_to_sd_sqrt2", "aad_share_of_sq"]].to_string(index=False,
                                                                       float_format=lambda v: f"{v:.5f}"))

    print("\n" + "=" * 78)
    print("2. REDUNDANCY  --  correlation of the composite with its components")
    print("=" * 78)
    print(red[["slice", "pearson_arin_sd", "pearson_arin_aad", "spearman_arin_sd",
               "spearman_arin_aad", "spearman_aad_sd", "r2_arin_from_sd"]].to_string(
        index=False, float_format=lambda v: f"{v:.4f}"))

    print("\n" + "=" * 78)
    print("3. RANK AGREEMENT  --  Kendall tau vs the ARIn ranking (1.0 = identical)")
    print("=" * 78)
    tau_cols = [c for c in rank.columns if c.startswith("tau_vs_")]
    print(rank[["slice", "grouping", "n_levels"] + tau_cols].to_string(
        index=False, float_format=lambda v: f"{v:.3f}"))

    print("\n  Mean tau across all slices/groupings:")
    for c in tau_cols:
        print(f"    {c:22s} {rank[c].mean():.3f}")

    print("\n" + "=" * 78)
    print("4. DECISION FLIPS  --  most-robust level under each aggregator")
    print("=" * 78)
    best_cols = [c for c in rank.columns if c.startswith("best_")]
    print(rank[["slice", "grouping"] + best_cols].to_string(index=False))

    flips = (rank["best_arin"] != rank["best_arin_vec"]).sum()
    print(f"\n  ARIn winner differs from TOPSIS-normalised winner in "
          f"{flips}/{len(rank)} groupings ({100 * flips / len(rank):.0f}%)")
    same_sd = (rank["best_arin"] == rank["best_sd_only"]).sum()
    print(f"  ARIn winner identical to SD-alone winner in "
          f"{same_sd}/{len(rank)} groupings ({100 * same_sd / len(rank):.0f}%)")

    if args.out_dir:
        dom.to_csv(os.path.join(args.out_dir, "arin_scale_dominance.csv"), index=False)
        red.to_csv(os.path.join(args.out_dir, "arin_redundancy.csv"), index=False)
        rank.to_csv(os.path.join(args.out_dir, "arin_rank_agreement.csv"), index=False)
        print(f"\n[saved] CSVs written to {args.out_dir}")


if __name__ == "__main__":
    main()
