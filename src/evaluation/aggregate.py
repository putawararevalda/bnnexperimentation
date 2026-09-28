"""Reduce per-fold SEU result CSVs into mean +/- std tables for the paper.

Metric definitions follow src/evaluation/metrics.py:
    AAD  = |accuracy_after - accuracy_before|
    ARIn = sqrt((AAD^2 + SoftmaxDiff^2) / 2)

Two aggregation layers live here and they answer different questions:

- ``aggregate_across_folds`` spreads over *injection rows*. Use it for
  descriptive tables only.
- ``fold_ci`` / ``paired_fold_difference`` spread over *folds*, which is the
  unit of analysis fixed by docs/revision/results_presentation_plan.html and
  the only one that answers Reviewer #1's request for confidence intervals.
"""
import glob
import logging
import os

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)

METRICS = ["aad", "softmax_difference", "arin", "initial_accuracy"]

GROUP_PRESETS: dict[str, list[str]] = {
    # Table 1 style: one row per model variant, averaged over everything else.
    "variant": ["variant"],
    # Per design-choice breakdown.
    "design": ["variant", "activation_fn", "prior", "prior_b"],
    # Full per-injection-site breakdown (Tables 3-5 style).
    "site": ["variant", "activation_fn", "prior", "prior_b",
             "param_type", "location_layer", "location_module", "bit_index"],
}

# Columns that jointly identify one injection site within a fold. Two folds are
# comparable only if they cover exactly this set of sites.
GRID_KEY: list[str] = [
    "activation_fn", "prior", "prior_b", "param_type",
    "location_layer", "location_module", "bit_index",
]

# Single-factor marginals for the revision's Table 1 and the bit/site figure.
# Each value is the set of columns a fold mean is taken within.
FACTOR_PRESETS: dict[str, list[str]] = {
    "overall": [],
    "activation": ["activation_fn"],
    "prior": ["prior"],
    "prior_scale": ["prior_b"],
    "variant": ["variant"],
    "layer": ["location_layer"],
    "site": ["location_layer", "location_module"],
    "bit": ["bit_index"],
    "param_type": ["param_type"],
    "orig_bit": ["original_bit_condition"],
    "layer_bit": ["location_layer", "location_module", "bit_index"],
}


def load_seu_results(root: str) -> pd.DataFrame:
    """Recursively load every SEU CSV under root into one DataFrame.

    Raises:
        FileNotFoundError: If no CSVs are found.
    """
    paths = sorted(glob.glob(os.path.join(root, "**", "*.csv"), recursive=True))
    if not paths:
        raise FileNotFoundError(f"no SEU CSVs found under {root}")

    frames = []
    for p in paths:
        df = pd.read_csv(p)
        df["source_file"] = os.path.relpath(p, root)
        frames.append(df)
    combined = pd.concat(frames, ignore_index=True)
    logger.info("Loaded %d rows from %d files", len(combined), len(paths))
    return combined


def add_robustness_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Add aad and arin columns. NaN inputs propagate to NaN outputs."""
    out = df.copy()
    out["aad"] = out["accuracy_change"].abs()
    out["arin"] = np.sqrt((out["aad"] ** 2 + out["softmax_difference"] ** 2) / 2)
    return out


def aggregate_across_folds(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    """Reduce to mean/std per group, excluding degenerate (NaN) injections.

    The reported std spreads over **injection rows**, not folds, despite the
    function name. It describes how much sites differ from one another, so it
    must not be turned into a confidence interval for a training-run effect --
    use fold_ci for that.

    Degenerate injections are those the SEU script marked NaN (negative
    scale/width). They are excluded from statistics and counted in n_excluded
    rather than silently dropped.

    Raises:
        KeyError: If any group column is absent.
    """
    missing = [c for c in group_cols if c not in df.columns]
    if missing:
        raise KeyError(f"group columns not in DataFrame: {missing}")

    valid = df["arin"].notna()
    records = []
    for keys, group in df.groupby(group_cols, dropna=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        good = group[valid.loc[group.index]]
        record = dict(zip(group_cols, keys))
        for metric in METRICS:
            if metric not in group.columns:
                continue
            series = good[metric].dropna()
            record[f"{metric}_mean"] = series.mean() if len(series) else np.nan
            record[f"{metric}_std"] = series.std(ddof=1) if len(series) > 1 else 0.0
        record["n_folds"] = int(good["fold"].nunique()) if "fold" in good.columns else len(good)
        record["n_excluded"] = int(len(group) - len(good))
        records.append(record)

    return pd.DataFrame(records)


def assert_grid_consistency(df: pd.DataFrame, grid_key: list[str] | None = None) -> int:
    """Verify every fold covers an identical set of injection sites.

    A per-fold mean is only comparable across folds if each fold averaged over
    the same sites. Two checks run: the full site grid must match, and the
    subset surviving degenerate-injection exclusion must match too -- a fold
    that excluded different sites would contribute a mean over a different
    population.

    Args:
        df: SEU rows carrying a "fold" column and the grid key columns.
        grid_key: Site-identifying columns. Defaults to GRID_KEY, restricted to
            those actually present.

    Returns:
        Number of unique injection sites per fold.

    Raises:
        KeyError: If the fold column is absent.
        ValueError: If any fold's site set differs from the first fold's.
    """
    if "fold" not in df.columns:
        raise KeyError('no "fold" column; fold-level CIs need per-fold results')

    key = [c for c in (grid_key or GRID_KEY) if c in df.columns]
    folds = sorted(df["fold"].unique())
    valid = df["arin"].notna()

    def site_sets(frame: pd.DataFrame) -> dict[int, set]:
        return {f: set(map(tuple, frame.loc[frame["fold"] == f, key].values))
                for f in folds}

    for label, sets in (("full", site_sets(df)), ("valid", site_sets(df[valid]))):
        reference = sets[folds[0]]
        for f in folds[1:]:
            if sets[f] == reference:
                continue
            missing = len(reference - sets[f])
            extra = len(sets[f] - reference)
            raise ValueError(
                f"{label} injection grid of fold {f} differs from fold "
                f"{folds[0]}: {missing} sites missing, {extra} unexpected. "
                "Fold means are not comparable until the grids match."
            )

    n_sites = len(site_sets(df)[folds[0]])
    logger.info("Grid consistency OK: %d folds x %d injection sites on %s",
                len(folds), n_sites, key)
    return n_sites


def fold_means(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    """Collapse each (group, fold) to one observation per metric.

    This is the reduction that makes the fold the unit of analysis: whatever
    the injection grid size, a fold contributes exactly one number.

    Raises:
        KeyError: If the fold column or any group column is absent.
    """
    if "fold" not in df.columns:
        raise KeyError('no "fold" column; fold-level CIs need per-fold results')
    missing = [c for c in group_cols if c not in df.columns]
    if missing:
        raise KeyError(f"group columns not in DataFrame: {missing}")

    metrics = [m for m in METRICS if m in df.columns]
    by = list(group_cols) + ["fold"]
    # mean() skips the NaNs the SEU script wrote for degenerate injections.
    return df.groupby(by, dropna=False)[metrics].mean().reset_index()


def _t_interval(values: np.ndarray, alpha: float) -> tuple[float, float, int]:
    """Return (mean, half-width, n) for a two-sided Student-t interval."""
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    n = len(v)
    if n == 0:
        return np.nan, np.nan, 0
    if n == 1:
        return float(v[0]), np.nan, 1
    crit = float(stats.t.ppf(1.0 - alpha / 2.0, n - 1))
    return float(v.mean()), crit * float(v.std(ddof=1)) / np.sqrt(n), n


def fold_ci(df: pd.DataFrame, group_cols: list[str],
            alpha: float = 0.05) -> pd.DataFrame:
    """Marginal per-level mean with a Student-t CI over folds.

    One fold is one observation, so at k=5 the interval has 4 degrees of
    freedom and t(0.975, 4) = 2.776. Intervals are deliberately NOT taken over
    injection sites: the grid is an exhaustive deterministic sweep, not a
    sample, and its spread measures site heterogeneity rather than the
    training-run variability Reviewer #1 asked about.

    Args:
        df: SEU rows with robustness metrics and a fold column.
        group_cols: Factor columns. Empty list gives one overall row.
        alpha: Two-sided significance level.

    Returns:
        One row per level with <metric>_{mean,ci95,lo,hi} columns, n_folds, and
        n_excluded (degenerate injections dropped from that level).
    """
    per_fold = fold_means(df, group_cols)
    metrics = [m for m in METRICS if m in per_fold.columns]

    excluded = (
        df.assign(_bad=df["arin"].isna())
          .groupby(group_cols, dropna=False)["_bad"].sum()
        if group_cols else pd.Series({(): int(df["arin"].isna().sum())})
    )

    records = []
    groups = (per_fold.groupby(group_cols, dropna=False) if group_cols
              else [((), per_fold)])
    for keys, group in groups:
        keys = keys if isinstance(keys, tuple) else (keys,)
        record = dict(zip(group_cols, keys))
        for metric in metrics:
            mean, half, n = _t_interval(group[metric].values, alpha)
            record[f"{metric}_mean"] = mean
            record[f"{metric}_ci95"] = half
            record[f"{metric}_lo"] = mean - half
            record[f"{metric}_hi"] = mean + half
        # Count only folds that produced a usable observation. A fold whose
        # every injection was degenerate averages to NaN and is dropped by
        # _t_interval, so counting it here would overstate the sample.
        usable = group["arin"].notna() if "arin" in group.columns else slice(None)
        record["n_folds"] = int(group.loc[usable, "fold"].nunique())
        lookup = keys[0] if len(keys) == 1 else keys
        record["n_excluded"] = int(excluded.get(lookup, 0))
        records.append(record)

    out = pd.DataFrame(records)
    if "arin_mean" in out.columns:
        out = out.sort_values("arin_mean").reset_index(drop=True)
    return out


def paired_fold_difference(df: pd.DataFrame, factor: str, level_a: str,
                           level_b: str, alpha: float = 0.05) -> pd.DataFrame:
    """CI on the per-fold difference (level_a - level_b) of one factor.

    Marginal intervals overlap at n=5 because folds differ in difficulty even
    when every fold agrees on the sign of an effect. That fold effect is shared
    by both levels and cancels in the difference, so two-condition claims
    (BNN vs DNN, WG vs ReLU) must be made this way.

    Returns:
        One row per metric with the mean difference, its CI, the number of
        paired folds, and n_agree -- how many folds share the sign of the mean.
        A CI excluding zero with n_agree < n_folds warrants a second look.

    Raises:
        KeyError: If factor is absent.
        ValueError: If either level is missing or no fold holds both.
    """
    if factor not in df.columns:
        raise KeyError(f"factor {factor!r} not in DataFrame")

    per_fold = fold_means(df, [factor]).set_index([factor, "fold"])
    for level in (level_a, level_b):
        if level not in per_fold.index.get_level_values(factor):
            raise ValueError(
                f"level {level!r} not found in {factor!r}; available: "
                f"{sorted(per_fold.index.get_level_values(factor).unique())}"
            )

    a, b = per_fold.xs(level_a), per_fold.xs(level_b)
    common = sorted(a.index.intersection(b.index))
    if not common:
        raise ValueError(f"no fold contains both {level_a!r} and {level_b!r}")
    if len(common) < len(a) or len(common) < len(b):
        logger.warning("%s vs %s: pairing on %d folds, dropping unmatched ones",
                       level_a, level_b, len(common))

    records = []
    for metric in [m for m in METRICS if m in a.columns]:
        diff = (a.loc[common, metric] - b.loc[common, metric]).values
        mean, half, n = _t_interval(diff, alpha)
        records.append({
            "factor": factor,
            "contrast": f"{level_a} - {level_b}",
            "metric": metric,
            "mean_diff": mean,
            "ci95": half,
            "lo": mean - half,
            "hi": mean + half,
            "n_folds": n,
            "n_agree": int(np.sum(np.sign(diff) == np.sign(mean))),
            "excludes_zero": bool(np.isfinite(half) and abs(mean) > half),
        })
    return pd.DataFrame(records)


def format_ci_column(df: pd.DataFrame, metric: str, digits: int = 4) -> pd.Series:
    """Render "<mean> $\\pm$ <half-width>" for a LaTeX table cell."""
    mean, half = df[f"{metric}_mean"], df[f"{metric}_ci95"]
    return pd.Series([
        f"{m:.{digits}f}" if not np.isfinite(h) else f"{m:.{digits}f} $\\pm$ {h:.{digits}f}"
        for m, h in zip(mean, half)
    ], index=df.index)


def latex_escape(values: pd.Series) -> pd.Series:
    """Escape LaTeX specials in a column of labels.

    For use with to_latex(escape=False), where cells already carry intentional
    markup such as $\\pm$ and so cannot be escaped wholesale.
    """
    replacements = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%",
                    "$": r"\$", "#": r"\#", "_": r"\_", "{": r"\{",
                    "}": r"\}", "~": r"\textasciitilde{}",
                    "^": r"\textasciicircum{}"}
    def escape_one(value) -> str:
        text = str(value)
        for old, new in replacements.items():
            text = text.replace(old, new)
        return text
    return values.map(escape_one)


def to_latex(df: pd.DataFrame, caption: str, label: str, float_fmt: str = "%.4f",
             escape: bool = True) -> str:
    """Render a LaTeX table wrapped in resizebox to fit ICAART column width.

    Reviewer #1 flagged Tables 3-5 as running out of page bounds, so width is
    constrained rather than left to the default.

    Args:
        escape: Escape LaTeX specials in every cell. Pass False when cells hold
            deliberate markup (see format_ci_column) and escape the label
            columns yourself with latex_escape.
    """
    body = df.to_latex(index=False, float_format=float_fmt, escape=escape,
                       longtable=False)
    return (
        "\\begin{table}[t]\n"
        "\\centering\n"
        f"\\caption{{{caption}}}\n"
        f"\\label{{{label}}}\n"
        "\\resizebox{\\columnwidth}{!}{%\n"
        f"{body}"
        "}\n"
        "\\end{table}\n"
    )
