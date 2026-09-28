"""
Train Bayesian CNN on EuroSAT dataset (10-class).

Sweeps over prior distributions x activation functions x prior scale (b) values.
Runs one of four model variants via --variant flag.

Usage examples:
    python scripts/train_eurosat.py --variant 00 --epoch 100
    python scripts/train_eurosat.py --variant 01 --epoch 100 --prior Laplace_prior
    python scripts/train_eurosat.py --variant 00 --trial-mode --epoch 10
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

from src.data.eurosat import load_data
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
    parser = argparse.ArgumentParser(description="Train Bayesian CNN on EuroSAT (full sweep)")
    parser.add_argument("--variant", type=str, default="00", choices=["00", "01", "02", "03"],
                        help="Model variant: 00=base 01=smartpool 02=dropout 03=weight_decay")
    parser.add_argument("--prior", type=str, default="all",
                        choices=["Gaussian_prior", "Laplace_prior", "Uniform_prior", "all"],
                        help="Prior distribution. Default: all (sweep all three)")
    parser.add_argument("--activation", type=str, default=None,
                        choices=["relu", "tanh", "sigmoid", "sinusoidal", "relu6", "wg", "rwg"],
                        help="Run only this activation function. Default: all 7")
    parser.add_argument("--epoch", type=int, default=100,
                        help="Number of training epochs. Default: 100")
    parser.add_argument("--b-set", type=str, default="full", choices=["full", "single"],
                        help="Prior scale sweep: full=[10.0,1.0,0.1], single=[1.0]. Default: full")
    parser.add_argument("--b-value", type=float, default=None,
                        help="Run only this specific b value (e.g. 10.0). Overrides --b-set.")
    parser.add_argument("--run-tag", type=str, default="paper_final",
                        help="MLflow run_set tag for this batch (e.g. eurosat-precheck-50epoch). Default: paper_final")
    parser.add_argument("--save-dir", type=str, default=None,
                        help="Override save directory (default: results/eurosat/bayesian/results_eurosat_v02_{variant})")
    parser.add_argument("--num-particles", type=int, default=1,
                        help="Number of ELBO particles (gradient estimate quality). Default: 1 (legacy)")
    parser.add_argument("--init-scale", type=float, default=None,
                        help="Override guide init_scale (default: 0.25*b, legacy). Try 0.01.")
    parser.add_argument("--scale-likelihood", dest="scale_likelihood", action="store_true",
                        help="Scale the likelihood by N/batch_size for a correct minibatch ELBO "
                             "(fixes KL over-weighting). Default: off (legacy behavior).")
    parser.set_defaults(scale_likelihood=False)
    parser.add_argument("--resume", dest="resume", action="store_true",
                        help="Skip combinations already listed in {save_dir}/completed_runs.txt "
                             "(written after each combo finishes). Safe to rerun the same "
                             "command after an interruption.")
    parser.set_defaults(resume=False)
    parser.add_argument("--trial-mode", dest="trial_mode", action="store_true",
                        help="Run only 1 combination (restricts activations/priors/b, keeps --epoch as-is)")
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
    save_dir = args.save_dir or f"results/eurosat/bayesian/results_eurosat_v02_{args.variant}"
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

    combos = [(a, p, b) for p in prior_list for a in activation_list for b in b_list]
    total = len(combos)
    print(f"Variant: {args.variant} ({variant_cfg['label']})")
    print(f"Total combinations: {total}  epochs={args.epoch}  save_dir={save_dir}")

    done_file = os.path.join(save_dir, "completed_runs.txt")
    done = set()
    if args.resume and os.path.exists(done_file):
        with open(done_file) as f:
            done = {line.strip() for line in f if line.strip()}
        print(f"Resume mode: {len(done)} combination(s) already completed will be skipped")

    try:
        import mlflow
        mlflow.set_experiment("bnn-seu-eurosat")
    except Exception:
        pass

    for exp_num, (act, prior_dist, b) in enumerate(combos, start=1):
        combo_key = f"{act}_{prior_dist}_b{b}_ep{args.epoch}"
        # Re-read the marker file before each combo so parallel windows
        # sharing the same save_dir see each other's completions.
        if args.resume and os.path.exists(done_file):
            with open(done_file) as f:
                done = {line.strip() for line in f if line.strip()}
        if combo_key in done:
            print(f"[resume] {exp_num}/{total} skipping {combo_key} (already completed)")
            continue

        send_telegram_message(
            title=f"EuroSAT v02_{args.variant} {exp_num}/{total}",
            message=f"activation={act}, prior={prior_dist}, b={b}"
        )

        pyro.clear_param_store()
        t_start = time.time()

        model = BayesShipsCNN(
            num_classes=10, device=device,
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

        train_loader, test_loader = load_data(batch_size=54)

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
            _mlflow.set_tags({
                "run_set": args.run_tag,
                "model_variant": f"v02_{args.variant}",
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

        # Mark combo as completed (always written, so a later --resume can pick up)
        with open(done_file, "a") as f:
            f.write(combo_key + "\n")

        elapsed = time.time() - t_start
        send_telegram_message(
            title=f"EuroSAT v02_{args.variant} {exp_num}/{total} Done",
            message=f"activation={act}, prior={prior_dist}, b={b}\n"
                    f"Test accuracy: {test_acc * 100:.2f}%\n"
                    f"Time: {elapsed:.1f}s"
        )


if __name__ == "__main__":
    main()
