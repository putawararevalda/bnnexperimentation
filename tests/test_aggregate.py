"""Tests for cross-fold SEU aggregation."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import pytest

from src.evaluation.aggregate import (
    add_robustness_metrics,
    aggregate_across_folds,
    to_latex,
)


def make_df() -> pd.DataFrame:
    """Two folds of one config; accuracy_change differs so std is non-zero."""
    rows = []
    for fold, acc_change, smd in [(1, -0.10, 0.20), (2, -0.20, 0.40)]:
        rows.append({
            "fold": fold, "variant": "base", "activation_fn": "relu",
            "prior": "gaussian", "prior_b": 1.0, "param_type": "locs",
            "location_layer": "conv1", "location_module": "weight", "bit_index": 0,
            "initial_accuracy": 0.90, "accuracy_change": acc_change,
            "softmax_difference": smd, "remarks": "",
        })
    return pd.DataFrame(rows)


def test_adds_aad_as_absolute_value():
    df = add_robustness_metrics(make_df())
    assert list(df["aad"]) == [0.10, 0.20]


def test_adds_arin_as_rms():
    df = add_robustness_metrics(make_df())
    expected = np.sqrt((0.10 ** 2 + 0.20 ** 2) / 2)
    assert abs(df["arin"].iloc[0] - expected) < 1e-9


def test_aggregates_mean_and_std_across_folds():
    df = add_robustness_metrics(make_df())
    out = aggregate_across_folds(df, ["variant", "activation_fn", "prior"])
    assert len(out) == 1
    row = out.iloc[0]
    assert abs(row["aad_mean"] - 0.15) < 1e-9
    assert abs(row["aad_std"] - np.std([0.10, 0.20], ddof=1)) < 1e-9
    assert row["n_folds"] == 2


def test_excludes_nan_rows_and_counts_them():
    df = add_robustness_metrics(make_df())
    df.loc[1, "softmax_difference"] = np.nan
    df.loc[1, "accuracy_change"] = np.nan
    df = add_robustness_metrics(df)
    out = aggregate_across_folds(df, ["variant"])
    row = out.iloc[0]
    assert row["n_folds"] == 1
    assert row["n_excluded"] == 1


def test_all_nan_group_yields_zero_folds_not_crash():
    df = add_robustness_metrics(make_df())
    df["accuracy_change"] = np.nan
    df["softmax_difference"] = np.nan
    df = add_robustness_metrics(df)
    out = aggregate_across_folds(df, ["variant"])
    assert out.iloc[0]["n_folds"] == 0
    assert out.iloc[0]["n_excluded"] == 2


def test_latex_output_contains_table_scaffolding():
    df = add_robustness_metrics(make_df())
    out = aggregate_across_folds(df, ["variant"])
    tex = to_latex(out, caption="Test", label="tab:test")
    assert "\\begin{table}" in tex
    assert "\\caption{Test}" in tex
    assert "\\label{tab:test}" in tex
    assert "resizebox" in tex, "must constrain width — reviewer flagged out-of-bounds tables"


def test_missing_group_column_raises():
    df = add_robustness_metrics(make_df())
    with pytest.raises(KeyError):
        aggregate_across_folds(df, ["nonexistent_column"])
