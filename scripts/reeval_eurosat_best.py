"""Re-evaluate EuroSAT test accuracy from the BEST checkpoint (MC-10).

The training script logs test accuracy using the LAST-epoch model, which at wide
priors (b=10) can diverge (see dropout/gaussian/relu/b=10: last-epoch 0.34 vs
best-checkpoint 0.77). This script reloads each config's saved best model +
param store and recomputes test accuracy the same way the SEU eval does, so the
accuracy table matches what SEU actually injects into.

Read-only w.r.t. training artifacts; writes one summary CSV. Resumable: rows
already in the output CSV are skipped.

Usage:
    uv run --with pyro-ppl,torch,torchvision,tqdm,scikit-learn,python-dotenv,requests,matplotlib,pandas,numpy \\
        python scripts/reeval_eurosat_best.py \\
        --root results/eurosat/bayesian --out results/tables_pilot/eurosat_accuracy_bestckpt.csv
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import glob
import json
import logging
import os
import re

import numpy as np
import pyro
import torch
from pyro.infer.autoguide import AutoNormal
from sklearn.metrics import confusion_matrix

from src.data.eurosat import load_data
from src.models.bayesian_cnn import BayesShipsCNN
from src.training.svi import predict_data
from src.utils.guide import AutoLaplace, AutoUniform

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

TS_RE = re.compile(r"(\d{8}_\d{6})")
DIR_TO_VARIANT = {
    "results_eurosat_v02_00": "base",
    "results_eurosat_v02_01": "smartpool",
    "results_eurosat_v02_02": "dropout",
    "results_eurosat_v02_03": "weight_decay",
}
GUIDE_BUILDERS = {"gaussian": AutoNormal, "laplace": AutoLaplace, "uniform": AutoUniform}

# EuroSAT configs store the activation function's __name__, not the constructor key.
ACT_NAME_TO_KEY = {"_actWG": "wg", "_actRWG": "rwg", "sin": "sinusoidal"}


def parse_args():
    p = argparse.ArgumentParser(description="Re-evaluate EuroSAT best checkpoints (MC-10)")
    p.add_argument("--root", type=str, default="results/eurosat/bayesian")
    p.add_argument("--out", type=str, default="results/tables_pilot/eurosat_accuracy_bestckpt.csv")
    p.add_argument("--num-samples", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=54)
    p.add_argument("--num-classes", type=int, default=10)
    return p.parse_args()


def find_artifact(variant_dir: str, kind: str, act: str, prior: str, ts: str) -> str | None:
    """kind in {'model','param_store'}. Match by act+prior+ts to avoid collisions."""
    token = f"_{act}_{prior}_"
    for path in glob.glob(os.path.join(variant_dir, f"{kind}_*{ts}*")):
        if token in os.path.basename(path):
            return path
    return None


def eval_one(config: dict, variant_dir: str, ts: str, test_loader,
             device, num_samples: int, num_classes: int) -> float:
    act, prior = config["activation"], config["prior"]
    b = config["prior_params"]["b"]
    mu = config["prior_params"]["mu"]

    # Files are named with the stored (function-name) activation; the model
    # constructor needs the canonical key.
    model_path = find_artifact(variant_dir, "model", act, prior, ts)
    param_path = find_artifact(variant_dir, "param_store", act, prior, ts)
    if not model_path or not param_path:
        raise FileNotFoundError(f"missing model/param for {act}/{prior} {ts}")

    act_key = ACT_NAME_TO_KEY.get(act, act)
    model = BayesShipsCNN(num_classes=num_classes, device=device,
                          activation=act_key, prior_dist=prior, mu=mu, b=b).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    guide = GUIDE_BUILDERS[prior](model, init_scale=0.05).to(device)
    pyro.clear_param_store()
    pyro.get_param_store().set_state(torch.load(param_path, weights_only=False))

    labels, preds = predict_data(model, guide, test_loader, device, num_samples=num_samples)
    cm = confusion_matrix(labels, preds)
    return float(np.trace(cm) / np.sum(cm))


def load_done(out_path: str) -> set:
    """Timestamps already evaluated (for resume)."""
    if not os.path.exists(out_path):
        return set()
    import pandas as pd
    try:
        return set(pd.read_csv(out_path)["ts"].astype(str))
    except (OSError, KeyError):
        return set()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, test_loader = load_data(batch_size=args.batch_size)
    logger.info("test set: %d imgs | device=%s", len(test_loader.dataset), device)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    done = load_done(args.out)
    if done:
        logger.info("resuming: %d configs already evaluated", len(done))

    header_needed = not os.path.exists(args.out)
    fh = open(args.out, "a", encoding="utf-8")
    if header_needed:
        fh.write("variant,prior,activation,b,ts,train_acc,test_acc_bestckpt\n")

    variant_dirs = sorted(d for d in glob.glob(os.path.join(args.root, "results_eurosat_v02_*"))
                          if os.path.basename(d) in DIR_TO_VARIANT)
    total = done_now = 0
    for vdir in variant_dirs:
        variant = DIR_TO_VARIANT[os.path.basename(vdir)]
        for cfg_path in sorted(glob.glob(os.path.join(vdir, "config_*.json"))):
            m = TS_RE.search(os.path.basename(cfg_path))
            if not m:
                continue
            ts = m.group(1)
            total += 1
            if ts in done:
                continue
            config = json.load(open(cfg_path))
            try:
                acc = eval_one(config, vdir, ts, test_loader, device,
                               args.num_samples, args.num_classes)
            except (FileNotFoundError, RuntimeError, KeyError) as e:
                logger.error("skip %s %s: %s", variant, ts, e)
                continue
            b = config["prior_params"]["b"]
            fh.write(f"{variant},{config['prior']},{config['activation']},"
                     f"{round(b, 2)},{ts},{config.get('best_accuracy')},{acc:.4f}\n")
            fh.flush()
            done_now += 1
            logger.info("[%d] %s %s/%s/b=%g -> best-ckpt test %.4f (was train-best %.4f)",
                        done_now, variant, config["prior"], config["activation"], b,
                        acc, config.get("best_accuracy") or -1)
    fh.close()
    logger.info("Done. Evaluated %d new configs (%d total seen). Output: %s",
                done_now, total, args.out)


if __name__ == "__main__":
    main()
