"""Fold-level confidence intervals for SEU robustness results.

The unit of analysis is the FOLD, not the injection row. Each fold is reduced
to one mean over its injection grid, and the interval is Student-t over those
k observations (t(0.975, 4) = 2.776 at k=5). An interval over the injection
grid would measure site-to-site heterogeneity, not training-run variability,
and would not answer Reviewer #1. See
docs/revision/results_presentation_plan.html.

Two-condition claims use --contrast, which pairs the two levels within each
fold so the shared fold effect cancels. At n=5 marginal intervals overlap even
when every fold agrees on the sign, so the paired form is the one to quote.

Usage:
    # Marginal table over every factor, ShipsNet base variant, folds 1-5
    uv run --with pandas,numpy,scipy python scripts/aggregate_seu_folds.py \\
        --root results/shipsnet/seu --glob "fold*/base/*.csv" \\
        --out results/tables/shipsnet_folds

    # One factor only, with LaTeX
    uv run --with pandas,numpy,scipy python scripts/aggregate_seu_folds.py \\
        --root results/shipsnet/seu --glob "fold*/base/*.csv" \\
        --preset activation --latex

    # Paired two-condition contrasts
    uv run --with pandas,numpy,scipy python scripts/aggregate_seu_folds.py \\
        --root results/shipsnet/seu --glob "fold*/base/*.csv" \\
        --contrast activation_fn:_actWG:relu \\
        --contrast prior:gaussian:laplace
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import glob as globmod
import logging
import re

import pandas as pd

from src.evaluation.aggregate import (
    FACTOR_PRESETS,
    add_robustness_metrics,
    assert_grid_consistency,
    fold_ci,
    format_ci_column,
    latex_escape,
    paired_fold_difference,
    to_latex,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

FOLD_RE = re.compile(r"fold(\d+)")

# Paper-ready column headers. Raw names carry underscores, which LaTeX would
# choke on in a non-escaped table.
HEADERS: dict[str, str] = {
    "activation_fn": "Activation",
    "prior": "Prior",
    "prior_b": "Prior scale $b$",
    "variant": "Variant",
    "location_layer": "Layer",
    "location_module": "Module",
    "bit_index": "Bit",
    "param_type": "Target",
    "original_bit_condition": "Orig.\\ bit",
}

# Caption wording per preset. Captions are typeset, so a raw preset name such
# as "layer_bit" would be a LaTeX error.
PRESET_TITLES: dict[str, str] = {
    "overall": "overall",
    "activation": "activation function",
    "prior": "prior distribution",
    "prior_scale": "prior scale $b$",
    "variant": "model variant",
    "layer": "layer",
    "site": "layer and parameter module",
    "bit": "attacked bit index",
    "param_type": "targeted parameter",
    "orig_bit": "original bit value",
    "layer_bit": "injection site and bit index",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fold-level CIs for SEU robustness results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Usage:")[1],
    )
    parser.add_argument("--root", type=str, default="results/shipsnet/seu",
                        help="directory the glob is resolved against")
    parser.add_argument("--glob", type=str, default="fold*/base/*.csv",
                        help="glob for the CSVs to include, relative to --root")
    parser.add_argument("--preset", type=str, default="all",
                        choices=["all"] + sorted(FACTOR_PRESETS),
                        help='factor to break down by ("all" runs every preset)')
    parser.add_argument("--contrast", type=str, action="append", default=[],
                        metavar="FACTOR:LEVEL_A:LEVEL_B",
                        help="paired per-fold difference; repeatable")
    parser.add_argument("--expect-folds", type=int, default=5,
                        help="warn if a level is missing folds; 0 disables")
    parser.add_argument("--alpha", type=float, default=0.05,
                        help="two-sided significance level")
    parser.add_argument("--out", type=str, default=None,
                        help="directory for CSV/LaTeX output; omit to print only")
    parser.add_argument("--latex", action="store_true",
                        help="also emit a LaTeX table per preset (needs --out)")
    parser.add_argument("--skip-grid-check", action="store_true",
                        help="proceed even if folds cover different sites "
                             "(makes intervals uncomparable; diagnostics only)")
    return parser.parse_args()


def load_folds(root: str, pattern: str) -> pd.DataFrame:
    """Load matching CSVs, deriving the fold index from each path.

    A "fold" column in the CSV wins; otherwise it is taken from a "foldN" path
    component, so both the newer per-fold schema and older layouts load.

    Raises:
        FileNotFoundError: If the glob matches nothing.
        ValueError: If a file has no fold column and no foldN path component.
    """
    paths = sorted(globmod.glob(str(Path(root) / pattern), recursive=True))
    if not paths:
        raise FileNotFoundError(f"no CSVs matched {pattern!r} under {root}")

    frames = []
    for path in paths:
        df = pd.read_csv(path)
        if "fold" not in df.columns:
            match = FOLD_RE.search(Path(path).as_posix())
            if not match:
                raise ValueError(
                    f"{path} has no 'fold' column and no foldN path component"
                )
            df["fold"] = int(match.group(1))
        df["source_file"] = str(Path(path).relative_to(root))
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    logger.info("Loaded %d rows from %d files, folds %s",
                len(combined), len(paths), sorted(combined["fold"].unique()))
    return combined


def render(table: pd.DataFrame, group_cols: list[str]) -> str:
    """Format a fold_ci result as an aligned text block."""
    display = table[group_cols].copy() if group_cols else pd.DataFrame(
        {"level": ["overall"]}, index=table.index)
    for metric, header in (("aad", "AAAD"), ("softmax_difference", "ASD"),
                           ("arin", "ARIn"), ("initial_accuracy", "Init acc")):
        if f"{metric}_mean" in table.columns:
            display[f"{header} [95% CI]"] = format_ci_column(table, metric)
    display["folds"] = table["n_folds"]
    display["excl"] = table["n_excluded"]
    return display.to_string(index=False).replace("$\\pm$", "+/-")


def run_presets(df: pd.DataFrame, args: argparse.Namespace,
                out_dir: Path | None) -> None:
    names = sorted(FACTOR_PRESETS) if args.preset == "all" else [args.preset]
    for name in names:
        group_cols = [c for c in FACTOR_PRESETS[name] if c in df.columns]
        if FACTOR_PRESETS[name] and not group_cols:
            logger.warning("Skipping preset %r: none of its columns are present", name)
            continue

        table = fold_ci(df, group_cols, alpha=args.alpha)
        print(f"\n=== {name} (by {group_cols or 'nothing, single row'}) ===")
        print(render(table, group_cols))

        if args.expect_folds:
            short = table[table["n_folds"] < args.expect_folds]
            for _, row in short.iterrows():
                level = ", ".join(str(row[c]) for c in group_cols) or "overall"
                logger.warning("Preset %r level %r has %d/%d folds -- its "
                               "interval is not comparable to the others",
                               name, level, row["n_folds"], args.expect_folds)

        if out_dir is not None:
            table.to_csv(out_dir / f"foldci_{name}.csv", index=False)
            if args.latex:
                # Cells carry deliberate $\pm$ markup, so escaping is applied to
                # the label columns only (activation names contain underscores).
                display = pd.DataFrame(
                    {HEADERS.get(c, c): latex_escape(table[c]) for c in group_cols},
                    index=table.index,
                )
                for metric, header in (("aad", "AAAD"), ("softmax_difference", "ASD"),
                                       ("arin", "ARIn")):
                    display[header] = format_ci_column(table, metric)
                (out_dir / f"foldci_{name}.tex").write_text(
                    to_latex(
                        display,
                        escape=False,
                        caption=(f"SEU robustness by {PRESET_TITLES.get(name, name)}, mean over "
                                 f"{args.expect_folds or 'k'} folds with 95\\% "
                                 "confidence intervals (Student-$t$). Each fold "
                                 "contributes one observation, averaged over its "
                                 "full injection grid."),
                        label=f"tab:foldci_{name}",
                    ),
                    encoding="utf-8",
                )


def run_contrasts(df: pd.DataFrame, args: argparse.Namespace,
                  out_dir: Path | None) -> None:
    if not args.contrast:
        return
    results = []
    for spec in args.contrast:
        parts = spec.split(":")
        if len(parts) != 3:
            raise ValueError(
                f"--contrast expects FACTOR:LEVEL_A:LEVEL_B, got {spec!r}"
            )
        factor, level_a, level_b = parts
        results.append(paired_fold_difference(df, factor, level_a, level_b,
                                              alpha=args.alpha))

    combined = pd.concat(results, ignore_index=True)
    print("\n=== paired per-fold differences ===")
    show = combined.copy()
    for col in ("mean_diff", "ci95", "lo", "hi"):
        show[col] = show[col].map(lambda v: f"{v:+.4f}")
    print(show[["contrast", "metric", "mean_diff", "ci95", "lo", "hi",
                "n_folds", "n_agree", "excludes_zero"]].to_string(index=False))

    disagreeing = combined[(combined["excludes_zero"])
                           & (combined["n_agree"] < combined["n_folds"])]
    for _, row in disagreeing.iterrows():
        logger.warning("%s / %s: interval excludes zero but only %d/%d folds "
                       "agree on the sign", row["contrast"], row["metric"],
                       row["n_agree"], row["n_folds"])

    if out_dir is not None:
        combined.to_csv(out_dir / "foldci_contrasts.csv", index=False)


def main() -> None:
    args = parse_args()
    if args.latex and not args.out:
        raise SystemExit("--latex needs --out")

    df = add_robustness_metrics(load_folds(args.root, args.glob))

    try:
        n_sites = assert_grid_consistency(df)
        print(f"Grid consistency: OK ({n_sites} injection sites per fold)")
    except ValueError:
        if not args.skip_grid_check:
            logger.error("Grid consistency check FAILED. Fold means are over "
                         "different site populations, so any interval computed "
                         "from them is invalid. Pass --skip-grid-check only to "
                         "inspect the data, never to produce paper numbers.")
            raise
        logger.warning("Grid consistency check FAILED but --skip-grid-check "
                       "was passed; results below are diagnostics, not paper "
                       "numbers.", exc_info=True)

    n_degenerate = int(df["arin"].isna().sum())
    logger.info("Degenerate injections excluded: %d of %d rows (%.1f%%)",
                n_degenerate, len(df), 100 * n_degenerate / len(df))

    out_dir = None
    if args.out:
        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)

    run_presets(df, args, out_dir)
    run_contrasts(df, args, out_dir)

    if out_dir is not None:
        logger.info("Wrote output to %s", out_dir)


if __name__ == "__main__":
    main()
