"""
Train Bayesian CNN on EuroSAT binary subset (Forest vs. SeaLake).

Usage examples:
    uv run python scripts/train_eurosat_binary.py --epoch 100 --prior Gaussian_prior --activation relu --b-value 1.0
    uv run python scripts/train_eurosat_binary.py --trial-mode
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import time
import os

import numpy as np
import pyro
import torch
from pyro.infer import SVI, Trace_ELBO
from pyro.infer.autoguide import AutoNormal
from pyro.optim import ClippedAdam
from sklearn.metrics import confusion_matrix

from src.data.eurosat_binary import load_data
from src.models.bayesian_cnn import BayesShipsCNN
from src.training.svi import train_svi_with_stats, plot_training_results_with_stats, predict_data
from src.utils.guide import AutoLaplace, AutoUniform
from src.utils.notify import send_telegram_message

import pandas as pd


VARIANT_CONFIG = {
    "00": {"smartpool": False, "dropout": False, "weight_decay": False, "label": "base"},
    "01": {"smartpool": True,  "dropout": False, "weight_decay": False, "label": "smartpool"},
    "02": {"smartpool": False, "dropout": True,  "weight_decay": False, "label": "dropout"},
    "03": {"smartpool": False, "dropout": False, "weight_decay": True,  "label": "weight_decay"},
}


def parse_args():
    parser = argparse.ArgumentParser(description="Train Bayesian CNN on EuroSAT binary (Forest vs. SeaLake)")
    parser.add_argument("--variant", type=str, default="00", choices=["00", "01", "02", "03"],
                        help="Model variant: 00=base 01=smartpool 02=dropout 03=weight_decay")
    parser.add_argument("--prior", type=str, default="Gaussian_prior",
                        choices=["Gaussian_prior", "Laplace_prior", "Uniform_prior", "all"],
                        help="Prior distribution. Default: Gaussian_prior")
    parser.add_argument("--activation", type=str, default=None,
                        choices=["relu", "tanh", "sigmoid", "sinusoidal", "relu6", "wg", "rwg"],
                        help="Activation function. Default: all 7")
    parser.add_argument("--epoch", type=int, default=100,
                        help="Number of training epochs. Default: 100")
    parser.add_argument("--b-set", type=str, default="full", choices=["full", "single"],
                        help="Prior scale sweep: full=[10.0,1.0,0.1], single=[1.0]. Default: full")
    parser.add_argument("--b-value", type=float, default=None,
                        help="Run only this specific b value (e.g. 1.0). Overrides --b-set.")
    parser.add_argument("--run-tag", type=str, default="eurosat_binary",
                        help="MLflow run_set tag. Default: eurosat_binary")
    parser.add_argument("--save-dir", type=str, default=None,
                        help="Override save directory.")
    parser.add_argument("--num-particles", type=int, default=1,
                        help="Number of ELBO particles (gradient estimate quality). Default: 4")
    parser.add_argument("--init-scale", type=float, default=None,
                        help="Override guide init_scale (default: 0.25*b). Try 0.01 to fix posterior collapse.")
    parser.add_argument("--scale-likelihood", dest="scale_likelihood", action="store_true",
                        help="Scale the likelihood by N/batch_size for a correct minibatch ELBO "
                             "(fixes KL over-weighting). Default: off (legacy behavior).")
    parser.set_defaults(scale_likelihood=False)
    parser.add_argument("--trial-mode", dest="trial_mode", action="store_true",
                        help="Run only 1 combination for 1 epoch (quick smoke-test)")
    parser.set_defaults(trial_mode=False)
    return parser.parse_args()


def build_guide(prior_dist, model, b, device, init_scale=None):
    scale = init_scale if init_scale is not None else 0.25 * b
    if prior_dist == "gaussian":
        return AutoNormal(model, init_scale=scale).to(device)
    elif prior_dist == "laplace":
        return AutoLaplace(model, init_scale=scale).to(device)
    elif prior_dist == "uniform":
        return AutoUniform(model, init_scale=scale).to(device)
    raise ValueError(f"Unknown prior_dist: {prior_dist}")


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))

    variant_cfg = VARIANT_CONFIG[args.variant]
    save_dir = args.save_dir or f"results/eurosat_binary/bayesian/results_v{args.variant}"
    os.makedirs(save_dir, exist_ok=True)

    prior_map = {
        "Gaussian_prior": "gaussian",
        "Laplace_prior": "laplace",
        "Uniform_prior": "uniform",
    }
    prior_list = ["gaussian", "laplace", "uniform"] if args.prior == "all" else [prior_map[args.prior]]
    activation_list = [args.activation] if args.activation else ["relu", "tanh", "sigmoid", "sinusoidal", "relu6", "wg", "rwg"]
    if args.b_value is not None:
        b_list = [args.b_value]
    else:
        b_list = [10.0, 1.0, 0.1] if args.b_set == "full" else [1.0]

    if args.trial_mode:
        activation_list = activation_list[:1]
        b_list = b_list[:1]
        prior_list = prior_list[:1]
        args.epoch = 1

    combos = [(a, p, b) for p in prior_list for a in activation_list for b in b_list]
    total = len(combos)
    print(f"Variant: {args.variant} ({variant_cfg['label']})")
    print(f"Total combinations: {total}  epochs={args.epoch}  save_dir={save_dir}")

    try:
        import mlflow
        mlflow.set_experiment("bnn-seu-eurosat-binary")
    except Exception:
        pass

    for exp_num, (act, prior_dist, b) in enumerate(combos, start=1):
        send_telegram_message(
            title=f"EuroSAT-Binary v{args.variant} {exp_num}/{total}",
            message=f"activation={act}, prior={prior_dist}, b={b}"
        )

        pyro.clear_param_store()
        t_start = time.time()

        model = BayesShipsCNN(
            num_classes=2, device=device,
            activation=act, prior_dist=prior_dist,
            mu=0.0, b=b,
            smartpool_switch=variant_cfg["smartpool"],
            dropout_switch=variant_cfg["dropout"],
        )
        guide = build_guide(prior_dist, model, b, device, init_scale=args.init_scale)

        wd = 1e-4 if variant_cfg["weight_decay"] else 0.0
        optimizer = ClippedAdam({"lr": 1e-3, "weight_decay": wd})
        svi = SVI(model=model, guide=guide, optim=optimizer,
                  loss=Trace_ELBO(num_particles=args.num_particles))

        model.to(device)
        guide.to(device)

        train_loader, test_loader = load_data(batch_size=16)

        if args.scale_likelihood:
            model.obs_scale = len(train_loader.dataset) / train_loader.batch_size
            print(f"Likelihood scaling ON: obs_scale = {model.obs_scale:.1f}")

        (losses, accuracies, accuracy_epochs,
         loc_stats, scale_stats,
         best_model_path, best_guide_path, best_ps_path,
         ts) = train_svi_with_stats(
            model, guide, svi, train_loader, device,
            num_epochs=args.epoch,
            save_dir=save_dir,
        )

        act_name = model.activation_fn.__name__ if hasattr(model.activation_fn, "__name__") else str(model.activation_fn)
        plot_training_results_with_stats(
            losses, accuracies, accuracy_epochs,
            loc_stats, scale_stats,
            act_name, prior_dist, ts,
            save_dir=save_dir,
        )

        labels, preds = predict_data(model, guide, test_loader, device, num_samples=10)
        cm = confusion_matrix(labels, preds)
        test_acc = np.trace(cm) / np.sum(cm)
        print(f"Test accuracy: {test_acc * 100:.4f}%")

        try:
            import mlflow as _mlflow
            _mlflow.log_metric("test_acc", test_acc)
            _mlflow.set_tags({
                "run_set": args.run_tag,
                "model_variant": f"v{args.variant}",
                "dataset": "eurosat_binary",
                "classes": "Forest_vs_SeaLake",
                "obs_scale": f"{model.obs_scale:.1f}",
                "init_scale": str(args.init_scale),
                "num_particles": str(args.num_particles),
            })
            _mlflow.end_run()
        except Exception:
            pass

        pd.DataFrame({"True Label": labels, "Predicted Label": preds}).to_csv(
            os.path.join(save_dir, f"predictions_{act_name}_{prior_dist}_{ts}_{test_acc*100:.0f}.csv"),
            index=False,
        )

        elapsed = time.time() - t_start
        send_telegram_message(
            title=f"EuroSAT-Binary v{args.variant} {exp_num}/{total} Done",
            message=f"activation={act}, prior={prior_dist}, b={b}\n"
                    f"Test accuracy: {test_acc * 100:.2f}%\n"
                    f"Time: {elapsed:.1f}s"
        )


if __name__ == "__main__":
    main()
