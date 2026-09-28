"""
Re-evaluate ShipsNet test accuracy from the saved BEST checkpoints (fold 1).

Why this exists
---------------
The training grids under `results/shipsnet/bayesian/results_shipsnet_v02_*` were
produced before `scripts/train_shipsnet.py` was fixed to reload the
`*_epoch_best_*` artifacts before `predict_data`. Their logged `test_acc` (and
the accuracy encoded in the `predictions_*_NN.csv` filenames) therefore describes
the **last-epoch** model, while `scripts/eval_seu_shipsnet.py` loads the **best**
checkpoint. Accuracy tables and SEU baselines consequently refer to two
different models.

This script recomputes test accuracy from the best checkpoints using the same
MC-10 inference the SEU scripts use, so both can be quoted from one model.

It is strictly additive: it reads the existing artifacts and writes new files
under `--save-dir`. It never modifies or overwrites anything in `--search-dir`,
and refuses to clobber its own previous output unless `--overwrite` is passed.

MC-10 inference is stochastic. Use `--repeats N` (with `--seed`) to quantify
run-to-run spread so a recomputed-vs-old difference can be judged against noise
rather than assumed meaningful.

Usage
-----
    uv run --with pyro-ppl,torch,torchvision,tqdm,scikit-learn,pandas,numpy \\
        python scripts/eval_best_test_acc_shipsnet.py \\
        --search-dir results/shipsnet/bayesian/results_shipsnet_v02_00 \\
        --save-dir results/shipsnet/best_checkpoint_acc/v02_00 \\
        --repeats 3
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import json
import os
import random
import re
from typing import Any

import numpy as np
import pandas as pd
import pyro
import torch
from pyro.infer.autoguide import AutoNormal
from sklearn.metrics import confusion_matrix

from src.data.shipsnet import load_data
from src.models.bayesian_cnn import BayesShipsCNN
from src.training.svi import predict_data
from src.utils.guide import AutoLaplace, AutoUniform

# `config_<activation>_<prior>_<YYYYmmdd>_<HHMMSS>.json`
CONFIG_RE = re.compile(r"^config_(?P<tag>.+)_(?P<ts>\d{8}_\d{6})\.json$")

# ShipsNet configs record the activation function's __name__ with the leading
# underscore stripped ("actWG"/"actRWG"). `BayesShipsCNN`'s act_map accepts the
# canonical keys and the EuroSAT-style "_actWG"/"_actRWG" aliases, but not the
# ShipsNet spelling, so normalise here rather than widen the shared model API.
ACTIVATION_ALIASES: dict[str, str] = {
    "actWG": "wg",
    "actRWG": "rwg",
    "_actWG": "wg",
    "_actRWG": "rwg",
    "sin": "sinusoidal",
}

VARIANT_SWITCHES: dict[str, dict[str, bool]] = {
    "base": {"smartpool_switch": False, "dropout_switch": False},
    "smartpool": {"smartpool_switch": True, "dropout_switch": False},
    "dropout": {"smartpool_switch": False, "dropout_switch": True},
    # weight decay is an optimizer setting; the architecture is the base one.
    "weight_decay": {"smartpool_switch": False, "dropout_switch": False},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recompute ShipsNet test accuracy from best checkpoints."
    )
    parser.add_argument(
        "--search-dir", type=str,
        default="results/shipsnet/bayesian/results_shipsnet_v02_00",
        help="Directory holding config/model/guide/param_store artifacts.",
    )
    parser.add_argument(
        "--save-dir", type=str,
        default="results/shipsnet/best_checkpoint_acc/v02_00",
        help="Where to write the new CSV. Created if absent.",
    )
    parser.add_argument("--fold", type=int, default=1, choices=[1, 2, 3, 4, 5])
    parser.add_argument("--num-samples", type=int, default=10,
                        help="Monte Carlo samples per batch (paper uses 10).")
    parser.add_argument("--repeats", type=int, default=1,
                        help="Independent MC evaluations per config, to quantify noise.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Base seed; repeat i uses seed+i.")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true",
                        help="Allow replacing an existing output CSV.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Evaluate only the first N configs (smoke testing).")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    """Seed every RNG that MC inference draws from."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    pyro.set_rng_seed(seed)


def build_guide(prior_dist: str, model: torch.nn.Module, device: torch.device):
    """Mirror `eval_seu_shipsnet.NewInjector._build_guide`.

    `init_scale` only affects lazy initialisation; every variational parameter is
    subsequently overwritten by the saved param store, so it cannot change results.
    """
    if prior_dist == "gaussian":
        return AutoNormal(model, init_scale=0.05).to(device)
    if prior_dist == "laplace":
        return AutoLaplace(model, init_scale=0.05).to(device)
    if prior_dist == "uniform":
        return AutoUniform(model, init_scale=0.05).to(device)
    raise ValueError(f"Unsupported prior: {prior_dist}")


def discover_configs(search_dir: str) -> list[dict[str, Any]]:
    """Pair every config JSON with its `*_epoch_best_*` model/guide/param artifacts."""
    entries: list[dict[str, Any]] = []
    missing: list[str] = []

    for fname in sorted(os.listdir(search_dir)):
        match = CONFIG_RE.match(fname)
        if not match:
            continue
        tag, ts = match.group("tag"), match.group("ts")

        paths = {
            "model_path": os.path.join(search_dir, f"model_{tag}_epoch_best_{ts}.pth"),
            "guide_path": os.path.join(search_dir, f"guide_{tag}_epoch_best_{ts}.pth"),
            "param_path": os.path.join(search_dir, f"param_store_{tag}_epoch_best_{ts}.pkl"),
        }
        absent = [os.path.basename(p) for p in paths.values() if not os.path.exists(p)]
        if absent:
            missing.append(f"{fname}: missing {', '.join(absent)}")
            continue

        with open(os.path.join(search_dir, fname)) as fh:
            config = json.load(fh)

        entries.append({"tag": tag, "ts": ts, "config": config, **paths})

    if missing:
        raise FileNotFoundError(
            "Incomplete best-checkpoint artifacts:\n  " + "\n  ".join(missing)
        )
    return entries


def assert_fold(config: dict[str, Any], expected_fold: int, ts: str) -> None:
    """Refuse to evaluate a model against a test set it may have trained on."""
    actual = config.get("fold")
    if actual is None:
        raise ValueError(
            f"config for {ts} has no 'fold' key - run "
            f"scripts/backfill_config_metadata.py first"
        )
    if actual != expected_fold:
        raise ValueError(
            f"fold mismatch for {ts}: model trained on fold {actual}, "
            f"but --fold {expected_fold} was requested"
        )


def evaluate_entry(entry: dict[str, Any], test_loader, device: torch.device,
                   args: argparse.Namespace) -> dict[str, Any]:
    """Load one best checkpoint and run `--repeats` independent MC evaluations."""
    config = entry["config"]
    variant = config.get("variant", "base")
    if variant not in VARIANT_SWITCHES:
        raise ValueError(f"unknown variant '{variant}' in config for {entry['ts']}")
    switches = VARIANT_SWITCHES[variant]

    accuracies: list[float] = []
    for repeat in range(args.repeats):
        set_seed(args.seed + repeat)
        pyro.clear_param_store()

        activation = ACTIVATION_ALIASES.get(config["activation"], config["activation"])
        model = BayesShipsCNN(
            num_classes=2, device=device,
            activation=activation,
            prior_dist=config["prior"],
            mu=config["prior_params"]["mu"],
            b=config["prior_params"]["b"],
            **switches,
        ).to(device)
        model.load_state_dict(torch.load(entry["model_path"], map_location=device))

        guide = build_guide(config["prior"], model, device)

        # Deliberately NOT calling guide.load_state_dict(): an AutoGuide builds its
        # parameters lazily on first call, so a freshly constructed guide has no
        # matching keys yet. The param store is the authoritative source in any
        # case - predict_data traces the guide through the global store, so every
        # variational parameter comes from the load below. This mirrors
        # eval_seu_shipsnet.NewInjector exactly, which is what makes the result
        # directly comparable to the SEU initial_accuracy baseline.
        #
        # weights_only=False: the param store pickles constraint objects, which
        # torch>=2.6 rejects under its weights_only=True default.
        pyro.get_param_store().set_state(
            torch.load(entry["param_path"], map_location=device, weights_only=False)
        )

        labels, preds = predict_data(
            model, guide, test_loader, device, num_samples=args.num_samples
        )
        cm = confusion_matrix(labels, preds)
        accuracies.append(float(np.trace(cm) / np.sum(cm)))

    arr = np.array(accuracies)
    return {
        "timestamp": entry["ts"],
        "activation_fn": config["activation"],
        "prior": config["prior"],
        "prior_mu": config["prior_params"]["mu"],
        "prior_b": config["prior_params"]["b"],
        "variant": variant,
        "fold": config.get("fold"),
        # Train accuracy at the selected epoch, recorded during training. NOT a
        # test figure - src/training/svi.py selects checkpoints on train accuracy.
        "best_train_accuracy": config.get("best_accuracy"),
        "best_accuracy_at_epoch": config.get("best_accuracy_at_epoch"),
        "recomputed_test_acc": float(arr.mean()),
        "recomputed_test_acc_std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
        "recomputed_test_acc_min": float(arr.min()),
        "recomputed_test_acc_max": float(arr.max()),
        "n_repeats": args.repeats,
        "num_samples": args.num_samples,
        "seed": args.seed,
    }


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(args.save_dir, exist_ok=True)
    out_path = os.path.join(
        args.save_dir, f"best_checkpoint_test_acc_fold{args.fold}.csv"
    )
    if os.path.exists(out_path) and not args.overwrite:
        raise FileExistsError(
            f"{out_path} already exists; pass --overwrite to replace it."
        )

    entries = discover_configs(args.search_dir)
    for entry in entries:
        assert_fold(entry["config"], args.fold, entry["ts"])
    if args.limit is not None:
        entries = entries[: args.limit]

    print(f"Device: {device}")
    print(f"Found {len(entries)} best checkpoints in {args.search_dir}")
    print(f"MC samples={args.num_samples}, repeats={args.repeats}, seed={args.seed}")

    _, test_loader = load_data(
        batch_size=args.batch_size, fold=args.fold, num_workers=args.num_workers
    )

    rows: list[dict[str, Any]] = []
    for idx, entry in enumerate(entries, start=1):
        cfg = entry["config"]
        label = f"{cfg['activation']}/{cfg['prior']}/b={cfg['prior_params']['b']}"
        print(f"[{idx}/{len(entries)}] {label} ({entry['ts']})")
        row = evaluate_entry(entry, test_loader, device, args)
        spread = f" +/- {row['recomputed_test_acc_std'] * 100:.4f}" if args.repeats > 1 else ""
        print(f"    recomputed test acc: {row['recomputed_test_acc'] * 100:.4f}%{spread}")
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(f"\nWrote {len(df)} rows to {out_path}")


if __name__ == "__main__":
    main()
