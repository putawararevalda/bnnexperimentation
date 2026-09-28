"""
Quantify the Monte-Carlo noise floor in the SEU robustness metrics.

Motivation
----------
`initial_accuracy` (the pre-flip baseline) is computed ONCE per model config and
reused as the reference for all ~168 injections of that config. MC-10 inference
is unseeded, so that baseline carries sampling noise which enters every
`accuracy_change` in the config as a systematic offset rather than averaging out.

`softmax_difference` is affected more directly: it compares two independent MC-10
draws, so even a completely inert bit flip yields a non-zero value.

Method
------
Benign mantissa bits are used as a null channel. Flipping mantissa bit 21 changes
a weight by ~1e-4 relative; bit 10 by ~25%. If the measured AAAD/ASD were tracking
real damage, these would differ substantially. If they instead agree with each
other, the shared value is the measurement floor, not a physical effect.

Two independent diagnostics separate noise from signal:
  * agreement across mantissa bits of very different magnitude -> floor
  * |mean signed change| << mean |change| -> symmetric scatter, i.e. noise
    (a real destructive effect drives the signed mean toward the absolute mean)

Floor-removed metrics are reported as a SENSITIVITY CHECK only. ASD is an L-infinity
softmax distance; its noise does not subtract linearly, so the adjusted figures
indicate direction and rough magnitude, not corrected values.

Usage
-----
    uv run --with pandas,numpy python scripts/analyze_seu_noise_floor.py
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd

# FP32 layout: bit 0 sign, bits 1-8 exponent, bits 9-31 mantissa.
MANTISSA_BITS = [10, 15, 21]

# Approximate relative magnitude of flipping each mantissa bit, for the
# "different magnitude, same measured effect" argument.
BIT_RELATIVE_MAGNITUDE = {10: "~2.5e-1", 15: "~7.8e-3", 21: "~1.2e-4"}

SLICES: list[tuple[str, str, float | None]] = [
    ("ShipsNet fold 1 v02_00 (paper)", "results/shipsnet/seu/results_shipsnet_v02_00_SEU", None),
    ("EuroSAT v02_00 base", "results/eurosat/seu_clean/v02_00", 1.0),
    ("EuroSAT v02_01 smartpool", "results/eurosat/seu_clean/v02_01", 1.0),
    ("EuroSAT v02_02 dropout", "results/eurosat/seu_clean/v02_02", 1.0),
    ("EuroSAT v02_03 weight_decay", "results/eurosat/seu_clean/v02_03", 1.0),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SEU Monte-Carlo noise floor analysis")
    parser.add_argument("--out", type=str, default=None,
                        help="Optional CSV path for the per-slice summary.")
    return parser.parse_args()


def load_slice(pattern_dir: str, prior_b: float | None) -> pd.DataFrame:
    """Load every SEU CSV in a directory, keeping only valid injections."""
    files = sorted(glob.glob(os.path.join(pattern_dir, "*.csv")))
    if not files:
        raise FileNotFoundError(f"no CSVs in {pattern_dir}")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    if prior_b is not None:
        df = df[df["prior_b"] == prior_b]
    return df[df["accuracy_after_seu"].notna()].copy()


def aggregate(df: pd.DataFrame) -> tuple[float, float, float]:
    """Paper metrics: AAAD, ASD, and ARIn built from the aggregated pair."""
    aaad = df["accuracy_change"].abs().mean()
    asd = df["softmax_difference"].mean()
    return aaad, asd, float(np.sqrt((aaad**2 + asd**2) / 2))


def per_bit_table(df: pd.DataFrame) -> pd.DataFrame:
    """Per-bit AAAD/ASD plus the signed-vs-absolute noise diagnostic."""
    rows = []
    for bit in sorted(df["bit_index"].unique()):
        sub = df[df["bit_index"] == bit]
        aaad = sub["accuracy_change"].abs().mean()
        signed = sub["accuracy_change"].mean()
        rows.append({
            "bit": int(bit),
            "n": len(sub),
            "AAAD": aaad,
            "ASD": sub["softmax_difference"].mean(),
            "mean_signed": signed,
            # ~0 => symmetric scatter (noise); ~1 => systematic damage (signal)
            "signal_ratio": abs(signed) / aaad if aaad else np.nan,
        })
    return pd.DataFrame(rows)


def analyse(name: str, path: str, prior_b: float | None) -> dict[str, float | str | int]:
    df = load_slice(path, prior_b)
    aaad, asd, arin = aggregate(df)

    benign = df[df["bit_index"].isin(MANTISSA_BITS)]
    f_aaad, f_asd, _ = aggregate(benign)

    adj_aaad, adj_asd = aaad - f_aaad, asd - f_asd
    adj_arin = float(np.sqrt((adj_aaad**2 + adj_asd**2) / 2))

    print(f"\n{'=' * 78}\n{name}\n  {path}"
          + (f"   (prior_b={prior_b})" if prior_b is not None else "")
          + f"\n{'=' * 78}")
    print(f"valid injections: {len(df)}   configs: "
          f"{len(df[['activation_fn', 'prior', 'prior_b']].drop_duplicates())}")

    print("\nPer-bit breakdown (signal_ratio ~0 = noise, ~1 = real damage):")
    tbl = per_bit_table(df)
    tbl["rel_magnitude"] = tbl["bit"].map(BIT_RELATIVE_MAGNITUDE).fillna("-")
    print(tbl.to_string(index=False, float_format=lambda v: f"{v:.5f}"))

    print(f"\nHeadline        AAAD={aaad:.5f}  ASD={asd:.5f}  ARIn={arin:.5f}")
    print(f"Noise floor     AAAD={f_aaad:.5f}  ASD={f_asd:.5f}"
          f"   ({f_aaad / aaad * 100:.0f}% of AAAD, {f_asd / asd * 100:.0f}% of ASD)")
    print(f"Floor-removed   AAAD={adj_aaad:.5f}  ASD={adj_asd:.5f}  ARIn={adj_arin:.5f}"
          "   [sensitivity check only]")

    # Per-config floor spread: is the floor uniform, or driven by a few configs?
    cfg_floor = (
        benign.groupby(["activation_fn", "prior", "prior_b"])["accuracy_change"]
        .apply(lambda s: s.abs().mean())
    )
    print(f"Per-config AAAD floor: min={cfg_floor.min():.5f}  "
          f"median={cfg_floor.median():.5f}  max={cfg_floor.max():.5f}")

    return {
        "slice": name, "n_valid": len(df),
        "AAAD": aaad, "ASD": asd, "ARIn": arin,
        "floor_AAAD": f_aaad, "floor_ASD": f_asd,
        "floor_pct_of_AAAD": f_aaad / aaad * 100,
        "floor_pct_of_ASD": f_asd / asd * 100,
        "adj_AAAD": adj_aaad, "adj_ASD": adj_asd, "adj_ARIn": adj_arin,
        "cfg_floor_min": cfg_floor.min(), "cfg_floor_max": cfg_floor.max(),
    }


def main() -> None:
    args = parse_args()
    summary = [analyse(name, path, b) for name, path, b in SLICES if os.path.isdir(path)]

    df = pd.DataFrame(summary)
    print(f"\n{'=' * 78}\nCROSS-SLICE SUMMARY\n{'=' * 78}")
    print(df[["slice", "AAAD", "ASD", "ARIn", "floor_AAAD", "floor_ASD",
              "floor_pct_of_ASD", "adj_ARIn"]]
          .to_string(index=False, float_format=lambda v: f"{v:.5f}"))

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        df.to_csv(args.out, index=False)
        print(f"\nWrote summary to {args.out}")


if __name__ == "__main__":
    main()
