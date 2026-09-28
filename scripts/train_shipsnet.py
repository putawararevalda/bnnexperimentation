"""
Train Bayesian CNN on ShipsNet dataset.

Sweeps over prior distributions × activation functions × prior scale (b) values.
Saves model/guide/param-store artifacts and training CSVs to --save-dir.

Usage examples:
    uv run python scripts/train_shipsnet.py --prior Gaussian_prior --epoch 100
    uv run python scripts/train_shipsnet.py --prior Laplace_prior --smartpool --epoch 50
    uv run python scripts/train_shipsnet.py --prior Uniform_prior --dropout-mode --b-set single
    uv run python scripts/train_shipsnet.py --fold 3 --scale-likelihood --epoch 100
    uv run python scripts/train_shipsnet.py --trial-mode   # quick smoke-run (1 combo, 1 epoch)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import glob
import json
import time
import os

import numpy as np
import pyro
import torch
from pyro.infer import SVI, Trace_ELBO
from pyro.infer.autoguide import AutoNormal
from pyro.optim import ClippedAdam
from sklearn.metrics import confusion_matrix

from src.data.shipsnet import load_data
from src.models.bayesian_cnn import BayesShipsCNN
from src.training.svi import train_svi_with_stats, plot_training_results_with_stats, predict_data
from src.utils.guide import AutoLaplace, AutoUniform
from src.utils.notify import send_telegram_message

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description='Train Bayesian CNN on ShipsNet')

    parser.add_argument('--prior', type=str, default='Gaussian_prior',
                        choices=['Gaussian_prior', 'Laplace_prior', 'Uniform_prior'],
                        help='Prior distribution family. Default: Gaussian_prior')
    parser.add_argument('--epoch', type=int, default=100,
                        help='Number of training epochs. Default: 100')
    parser.add_argument('--b-set', type=str, default='full', choices=['full', 'single'],
                        help='Prior scale sweep: full=[10.0,1.0,0.1], single=[1.0]. Default: full')
    parser.add_argument('--save-dir', type=str, default='results/shipsnet/bayesian',
                        help='Directory to save artifacts. Default: results/shipsnet/bayesian')
    parser.add_argument('--smartpool', dest='smartpool', action='store_true',
                        help='Use SmartPool instead of MaxPool')
    parser.add_argument('--no-smartpool', dest='smartpool', action='store_false')
    parser.set_defaults(smartpool=False)

    parser.add_argument('--wd', dest='weight_decay', action='store_true',
                        help='Enable weight decay (wd=1e-4) in ClippedAdam')
    parser.add_argument('--no-wd', dest='weight_decay', action='store_false')
    parser.set_defaults(weight_decay=False)

    parser.add_argument('--dropout-mode', dest='dropout_mode', action='store_true',
                        help='Enable Dropout(p=0.5) after conv2')
    parser.add_argument('--no-dropout-mode', dest='dropout_mode', action='store_false')
    parser.set_defaults(dropout_mode=False)

    parser.add_argument('--scale-likelihood', dest='scale_likelihood', action='store_true',
                        help='Scale the likelihood by N/batch_size for a correct minibatch ELBO '
                             '(fixes KL over-weighting). Default: off (legacy behavior). '
                             'Folds 1-2 were trained WITHOUT this flag.')
    parser.add_argument('--no-scale-likelihood', dest='scale_likelihood', action='store_false')
    parser.set_defaults(scale_likelihood=False)

    parser.add_argument('--trial-mode', dest='trial_mode', action='store_true',
                        help='Run only 1 combination for 1 epoch (quick test)')
    parser.add_argument('--no-trial-mode', dest='trial_mode', action='store_false')
    parser.set_defaults(trial_mode=False)

    parser.add_argument('--fold', type=int, default=None, choices=[1, 2, 3, 4, 5],
                        help='Cross-validation fold (1-5). Omit for the legacy split.')
    parser.add_argument('--num-workers', type=int, default=2,
                        help='DataLoader worker processes. Use 0 to load in-process '
                             '(lower RAM / avoids WinError 1455 when co-running another job).')

    return parser.parse_args()


# Map the constructor activation key to the function __name__ stored in configs.
_ACT_TO_FNNAME = {
    'relu': 'relu', 'tanh': 'tanh', 'sigmoid': 'sigmoid', 'sinusoidal': 'sin',
    'relu6': 'relu6', 'wg': '_actWG', 'rwg': '_actRWG',
}


def config_already_trained(save_dir: str, act: str, prior_dist: str, b: float) -> bool:
    """Resume support: True if a completed config for this (activation, prior, b)
    already exists in save_dir. A config JSON is written only after a run finishes,
    so its presence means that cell is done."""
    fn_name = _ACT_TO_FNNAME.get(act, act)
    for path in glob.glob(os.path.join(save_dir, 'config_*.json')):
        try:
            cfg = json.load(open(path))
        except (OSError, json.JSONDecodeError):
            continue
        if (cfg.get('activation') == fn_name and cfg.get('prior') == prior_dist
                and abs(cfg.get('prior_params', {}).get('b', -1.0) - b) < 1e-6):
            return True
    return False


def resolve_variant(args) -> str:
    """Map the mutually exclusive variant flags to the canonical variant name."""
    if args.smartpool:
        return 'smartpool'
    if args.dropout_mode:
        return 'dropout'
    if args.weight_decay:
        return 'weight_decay'
    return 'base'


def build_guide(prior_dist, model, b, device):
    scale = 0.25 * b
    if prior_dist == 'gaussian':
        return AutoNormal(model, init_scale=scale).to(device)
    elif prior_dist == 'laplace':
        return AutoLaplace(model, init_scale=scale).to(device)
    elif prior_dist == 'uniform':
        return AutoUniform(model, init_scale=scale).to(device)
    raise ValueError(f"Unknown prior_dist: {prior_dist}")


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    prior_map = {
        'Gaussian_prior': 'gaussian',
        'Laplace_prior': 'laplace',
        'Uniform_prior': 'uniform',
    }
    prior_dist = prior_map[args.prior]

    activation_list = ['relu', 'tanh', 'sigmoid', 'sinusoidal', 'relu6', 'wg', 'rwg']
    b_list = [10.0, 1.0, 0.1] if args.b_set == 'full' else [1.0]

    if args.trial_mode:
        activation_list = activation_list[:1]
        b_list = b_list[:1]

    total = len(activation_list) * len(b_list)
    print(f"Configuration: prior={args.prior}, smartpool={args.smartpool}, "
          f"dropout={args.dropout_mode}, wd={args.weight_decay}, "
          f"epochs={args.epoch}, b_set={args.b_set}, "
          f"scale_likelihood={args.scale_likelihood}")
    print(f"Total combinations: {total}")

    for exp_num, (act, b) in enumerate(
            [(a, b) for a in activation_list for b in b_list], start=1):

        if config_already_trained(args.save_dir, act, prior_dist, b):
            print(f"[skip {exp_num}/{total}] {act}/{prior_dist}/b={b} already trained "
                  f"in {args.save_dir}")
            continue

        send_telegram_message(
            title=f"ShipsNet Experiment {exp_num}/{total}",
            message=f"activation={act}, prior={prior_dist}, b={b}"
        )

        pyro.clear_param_store()
        t_start = time.time()

        model = BayesShipsCNN(
            num_classes=2, device=device,
            activation=act, prior_dist=prior_dist,
            mu=0.0, b=b,
            smartpool_switch=args.smartpool,
            dropout_switch=args.dropout_mode,
        )
        guide = build_guide(prior_dist, model, b, device)

        wd = 1e-4 if args.weight_decay else 0.0
        optimizer = ClippedAdam({"lr": 1e-3, "weight_decay": wd})
        svi = SVI(model=model, guide=guide, optim=optimizer,
                  loss=Trace_ELBO(num_particles=1))

        model.to(device)
        guide.to(device)

        train_loader, test_loader = load_data(batch_size=16, fold=args.fold,
                                              num_workers=args.num_workers)

        if args.scale_likelihood:
            model.obs_scale = len(train_loader.dataset) / train_loader.batch_size
            print(f"Likelihood scaling ON: obs_scale = {model.obs_scale:.1f}")

        (losses, accuracies, accuracy_epochs,
         loc_stats, scale_stats,
         best_model_path, best_guide_path, best_ps_path,
         ts) = train_svi_with_stats(
            model, guide, svi, train_loader, device,
            num_epochs=args.epoch,
            save_dir=args.save_dir,
            # obs_scale goes in the config JSON, not just MLflow tags: it changes
            # the objective being optimised, so it must be recoverable from the
            # artifacts alone. Folds 1-2 predate this field and are unscaled (1.0).
            extra_config={
                'fold': args.fold if args.fold is not None else 1,
                'variant': resolve_variant(args),
                'obs_scale': model.obs_scale,
            },
        )

        act_name = model.activation_fn.__name__ if hasattr(model.activation_fn, '__name__') else str(model.activation_fn)
        plot_training_results_with_stats(
            losses, accuracies, accuracy_epochs,
            loc_stats, scale_stats,
            act_name, prior_dist, ts,
            save_dir=args.save_dir,
        )

        # Evaluate the BEST checkpoint, not the final-epoch weights. Training
        # continues past the best epoch, so `model`/`guide` and the global Pyro
        # param store are all at the last epoch by this point. The SEU scripts
        # load these same *_epoch_best_* artifacts, so evaluating them here keeps
        # accuracy tables and SEU baselines referring to one model.
        # The param-store load is the essential one: predict_data traces the
        # guide through the global store, so restoring only the module
        # state_dicts would leave the last-epoch variational parameters in play.
        if best_model_path and best_guide_path and best_ps_path:
            model.load_state_dict(torch.load(best_model_path, map_location=device))
            guide.load_state_dict(torch.load(best_guide_path, map_location=device))
            # weights_only=False: the param store pickles constraint objects
            # (torch.distributions.constraints._Real), which torch>=2.6 rejects
            # under its weights_only=True default. Same load the SEU scripts use.
            pyro.clear_param_store()
            pyro.get_param_store().set_state(
                torch.load(best_ps_path, map_location=device, weights_only=False))
            eval_source = "best"
        else:
            print("WARNING: no best checkpoint saved; evaluating last-epoch weights")
            eval_source = "last"

        labels, preds = predict_data(model, guide, test_loader, device, num_samples=10)
        cm = confusion_matrix(labels, preds)
        test_acc = np.trace(cm) / np.sum(cm)
        print(f"Test accuracy ({eval_source} checkpoint): {test_acc * 100:.4f}%")

        try:
            import mlflow as _mlflow
            _mlflow.log_metric("test_acc", test_acc)
            _mlflow.set_tags({"obs_scale": f"{model.obs_scale:.1f}"})
            _mlflow.end_run()
        except Exception:
            pass

        pd.DataFrame({'True Label': labels, 'Predicted Label': preds}).to_csv(
            os.path.join(args.save_dir, f'predictions_{act_name}_{prior_dist}_{ts}_{test_acc*100:.0f}.csv'),
            index=False
        )

        elapsed = time.time() - t_start
        send_telegram_message(
            title=f"ShipsNet Experiment {exp_num}/{total} Done",
            message=f"activation={act}, prior={prior_dist}, b={b}\n"
                    f"Test accuracy: {test_acc * 100:.2f}%\n"
                    f"Time: {elapsed:.1f}s"
        )


if __name__ == "__main__":
    main()
