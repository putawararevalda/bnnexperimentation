"""Aggregate ShipsNet SEU results across folds into paper-ready tables.

Usage:
    uv run --with pandas,numpy python scripts/aggregate_seu.py \\
        --root results/shipsnet/seu --preset variant --out results/tables
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import logging

from src.evaluation.aggregate import (
    GROUP_PRESETS,
    add_robustness_metrics,
    aggregate_across_folds,
    load_seu_results,
    to_latex,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Aggregate SEU results across folds")
    parser.add_argument("--root", type=str, default="results/shipsnet/seu")
    parser.add_argument("--preset", type=str, default="variant", choices=sorted(GROUP_PRESETS))
    parser.add_argument("--out", type=str, default="results/tables")
    parser.add_argument("--caption", type=str, default="SEU robustness, mean $\\pm$ std across 5 folds")
    parser.add_argument("--label", type=str, default="tab:seu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = load_seu_results(args.root)
    df = add_robustness_metrics(df)

    group_cols = GROUP_PRESETS[args.preset]
    agg = aggregate_across_folds(df, group_cols)

    incomplete = agg[agg["n_folds"] < 5]
    if len(incomplete):
        logger.warning("%d/%d groups have fewer than 5 folds", len(incomplete), len(agg))

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"seu_{args.preset}.csv"
    tex_path = out_dir / f"seu_{args.preset}.tex"

    agg.to_csv(csv_path, index=False)
    tex_path.write_text(to_latex(agg, caption=args.caption, label=args.label), encoding="utf-8")

    logger.info("Wrote %s and %s (%d groups)", csv_path, tex_path, len(agg))
    logger.info("Total degenerate injections excluded: %d", int(agg["n_excluded"].sum()))


if __name__ == "__main__":
    main()
