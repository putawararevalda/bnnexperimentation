"""
Inject Single Event Upsets (SEUs) into trained Bayesian CNN models on ShipsNet.

For each trained model found in --search-dir, iterates over:
  (layer × module × bit_index × guide_parameter) and records accuracy change,
  softmax difference, and absolute weight difference per flip.

Skips timestamps already present in --save-dir to allow resuming.

Usage examples:
    uv run python scripts/eval_seu_shipsnet.py --prior Gaussian_prior \\
        --search-dir results/shipsnet/bayesian \\
        --save-dir results/shipsnet/seu

    uv run python scripts/eval_seu_shipsnet.py --prior Laplace_prior --limited-mode \\
        --search-dir results/shipsnet/bayesian \\
        --save-dir results/shipsnet/seu
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import json
import os

import numpy as np
import pandas as pd
import pyro
import pyro.distributions as dist
import torch
import torch.nn.functional as F
from pyro.infer.autoguide import AutoNormal
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

from src.data.shipsnet import load_data
from src.evaluation.seu import bitflip_float32_with_original
from src.models.bayesian_cnn import BayesShipsCNN
from src.utils.guide import AutoLaplace, AutoUniform
from src.utils.notify import send_telegram_message


def parse_args():
    parser = argparse.ArgumentParser(description='SEU evaluation for Bayesian ShipsNet models')
    parser.add_argument('--prior', type=str, default='Gaussian_prior',
                        choices=['Gaussian_prior', 'Laplace_prior', 'Uniform_prior'])
    parser.add_argument('--save-dir', type=str, default='results/shipsnet/seu',
                        help='Where to save CSV results.')
    parser.add_argument('--search-dir', type=str, default='results/shipsnet/bayesian',
                        help='Directory containing trained model artifacts.')
    parser.add_argument('--limited-mode', dest='limited_mode', action='store_true',
                        help='Only evaluate models with prior scale b=1.0')
    parser.add_argument('--no-limited-mode', dest='limited_mode', action='store_false')
    parser.add_argument('--fast-smartpool', action='store_true',
                        help='Use SmartPoolFast (bit-identical to SmartPool, ~19x '
                             'faster pooling). Only affects smartpool variants.')
    parser.set_defaults(limited_mode=False)
    parser.add_argument('--fold', type=int, default=None, choices=[1, 2, 3, 4, 5],
                        help='Cross-validation fold (1-5) the models were trained on. '
                             'Must match the fold in the model configs.')
    return parser.parse_args()


class NewInjector:
    """Injects SEUs into Pyro guide parameters and measures accuracy/softmax impact."""

    def __init__(self, trained_model, device, test_loader, num_samples,
                 pyro_param_store_path):
        self.device = device
        self.trained_model = trained_model.to(device)
        self.test_loader = test_loader
        self.trained_model.eval()
        self.num_samples = num_samples
        self.pyro_param_store_path = pyro_param_store_path

        self._build_guide()
        pyro.clear_param_store()
        pyro.get_param_store().set_state(
            torch.load(pyro_param_store_path, weights_only=False)
        )

        initial_labels, initial_preds, _, initial_probs = self._predict_probs(num_samples)
        self.initial_accuracy = self._accuracy(initial_labels, initial_preds)
        self.initial_probs = np.array(initial_probs)
        print(f"Initial accuracy: {self.initial_accuracy:.4%}")

    def _build_guide(self):
        prior = self.trained_model.prior_dist
        if prior == 'gaussian':
            self.guide = AutoNormal(self.trained_model, init_scale=0.05).to(self.device)
        elif prior == 'laplace':
            self.guide = AutoLaplace(self.trained_model, init_scale=0.05).to(self.device)
        elif prior == 'uniform':
            self.guide = AutoUniform(self.trained_model, init_scale=0.05).to(self.device)
        else:
            raise ValueError(f"Unsupported prior: {prior}")

    def _predict_probs(self, num_samples):
        all_labels, all_preds, all_logits, all_probs = [], [], [], []
        with torch.no_grad():
            for images, labels in tqdm(self.test_loader, desc="Evaluating", leave=False):
                images, labels = images.to(self.device), labels.to(self.device)
                n_classes = self.trained_model.fc1.out_features
                logits_mc = torch.zeros(num_samples, images.size(0), n_classes, device=self.device)
                for i in range(num_samples):
                    trace = pyro.poutine.trace(self.guide).get_trace(images)
                    replayed = pyro.poutine.replay(self.trained_model, trace=trace)
                    logits_mc[i] = replayed(images)
                avg_logits = logits_mc.mean(0)
                preds = avg_logits.argmax(1)
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
                all_logits.extend(avg_logits.cpu().numpy())
                all_probs.extend(F.softmax(avg_logits, dim=1).cpu().numpy())
        return all_labels, all_preds, all_logits, all_probs

    def _accuracy(self, labels, preds):
        cm = confusion_matrix(labels, preds)
        return np.trace(cm) / np.sum(cm)

    def _softmax_diff(self, before_probs, after_logits, penalty=1.0):
        before = np.asarray(before_probs, dtype=np.float32)
        after_t = torch.from_numpy(np.asarray(after_logits, dtype=np.float32))
        N = after_t.shape[0]
        finite = torch.isfinite(after_t).all(1).numpy()
        safe = torch.zeros_like(after_t)
        if finite.any():
            safe[finite] = F.softmax(after_t[finite], dim=1)
        safe = safe.numpy()
        diffs = np.where(finite,
                         np.max(np.abs(before - safe), axis=1),
                         penalty)
        return float(diffs.mean())

    def run_seu(self, location_index, param_unique, parameter_name,
                layer, layer_module, bit_i, num_samples):
        assert parameter_name in ["locs", "scales", "lows", "widths"]
        assert 0 <= bit_i < 32
        remarks = ""
        param_key = f"{param_unique}.{parameter_name}.{layer}.{layer_module}"

        pyro.clear_param_store()
        pyro.get_param_store().set_state(
            torch.load(self.pyro_param_store_path, weights_only=False)
        )

        with torch.no_grad():
            param = pyro.get_param_store().get_param(param_key)
            flat = param.clone().view(-1)
            orig_val = flat[location_index].cpu().item()
            flipped, orig_bit = bitflip_float32_with_original(orig_val, bit_i)

            if parameter_name in ["widths", "scales"] and flipped < 0:
                return {"accuracy_change": np.nan - self.initial_accuracy,
                        "softmax_difference": np.nan, "absolute_difference": np.nan,
                        "remarks": "SEU scale/width negative", "original_bit_condition": orig_bit}

            if np.isnan(flipped) or np.isinf(flipped):
                remarks += "SEU NaN/Inf clipped; "
                flipped = np.nan_to_num(
                    flipped,
                    nan=np.finfo(np.float32).max if orig_val >= 0 else -np.finfo(np.float32).max,
                    posinf=np.finfo(np.float32).max,
                    neginf=-np.finfo(np.float32).max,
                )

            seu_tensor = torch.tensor(flipped, dtype=param.dtype, device=param.device)
            abs_diff = abs(orig_val - flipped)
            flat[location_index] = seu_tensor
            pyro.get_param_store().__setitem__(param_key, flat.view(param.shape))

            if param_unique == "AutoUniform":
                low_key = f"{param_unique}.lows.{layer}.{layer_module}"
                low_flat = pyro.get_param_store().get_param(low_key).view(-1)
                low_val = seu_tensor if parameter_name == "lows" else low_flat[location_index]
                delta = (torch.nextafter(low_val,
                                         torch.tensor(float("inf"), dtype=low_val.dtype,
                                                      device=low_val.device)) - low_val)
                width_key = f"{param_unique}.widths.{layer}.{layer_module}"
                width_param = pyro.get_param_store().get_param(width_key)
                wflat = width_param.clone().view(-1)
                orig_w = wflat[location_index].item()
                wflat[location_index] = torch.max(wflat[location_index], delta)
                if orig_w != wflat[location_index].item():
                    remarks += "AutoUniform width adjusted; "
                pyro.get_param_store().__setitem__(width_key, wflat.view(width_param.shape))

                if parameter_name == "widths" and seu_tensor < 0:
                    return {"accuracy_change": np.nan - self.initial_accuracy,
                            "softmax_difference": np.nan, "absolute_difference": np.nan,
                            "remarks": remarks + "SEU width negative", "original_bit_condition": orig_bit}

        print(f"  {param_key}[{location_index}]: {orig_val:.6g} -> {flipped:.6g} (diff {abs_diff:.3g})")

        self._build_guide()
        after_labels, after_preds, after_logits, _ = self._predict_probs(num_samples)
        accuracy_after = self._accuracy(after_labels, after_preds)
        softmax_diff = self._softmax_diff(self.initial_probs, after_logits)

        print(f"  Accuracy after SEU: {accuracy_after:.3%}")
        return {
            "accuracy_change": accuracy_after - self.initial_accuracy,
            "softmax_difference": softmax_diff,
            "absolute_difference": abs_diff,
            "remarks": remarks,
            "original_bit_condition": orig_bit,
        }


# Model variant -> architecture switches. SmartPool is ACTIVE at eval time, so a
# smartpool model MUST be rebuilt with it enabled; dropout is inert under .eval()
# and weight decay is an optimizer setting, so both share the base architecture.
# Neither SmartPool nor Dropout has learnable parameters, which is why rebuilding
# with the wrong switch still load_state_dict()s cleanly - the mismatch is silent.
VARIANT_SWITCHES: dict[str, dict[str, bool]] = {
    'base': {'smartpool_switch': False, 'dropout_switch': False},
    'smartpool': {'smartpool_switch': True, 'dropout_switch': False},
    'dropout': {'smartpool_switch': False, 'dropout_switch': True},
    'weight_decay': {'smartpool_switch': False, 'dropout_switch': False},
}


def variant_from_search_dir(search_dir: str) -> str | None:
    """Infer the variant from the search-dir name, or None if unrecognised."""
    name = os.path.basename(os.path.normpath(search_dir)).lower()
    for variant in ('smartpool', 'dropout', 'weight_decay'):
        if variant in name:
            return variant
    return None


def resolve_variant(config: dict, search_dir: str, timestamp: str) -> str:
    """Determine the model variant, preferring the config over the directory name.

    Legacy configs (pre-variant experiments) have no 'variant' key; those runs are
    all plain base models, but fall back to the directory name first so a variant
    directory of un-backfilled configs cannot be silently mistaken for base.
    """
    from_config = config.get('variant')
    from_dir = variant_from_search_dir(search_dir)

    if from_config is None:
        variant = from_dir or 'base'
        if from_dir:
            print(f"  WARNING: config for {timestamp} has no 'variant' key; "
                  f"using '{variant}' inferred from search-dir")
    else:
        variant = from_config
        if from_dir is not None and from_dir != from_config:
            raise ValueError(
                f"variant mismatch for {timestamp}: config says '{from_config}' "
                f"but search-dir '{search_dir}' implies '{from_dir}'"
            )

    if variant not in VARIANT_SWITCHES:
        raise ValueError(
            f"unknown variant '{variant}' for {timestamp}; "
            f"expected one of {sorted(VARIANT_SWITCHES)}"
        )
    return variant


def load_model(timestamp, search_dir, config_files, model_files, param_files, device,
               num_classes=2, fast_smartpool=False):
    config_path = os.path.join(search_dir, config_files[timestamp])
    model_path = os.path.join(search_dir, model_files[timestamp])
    param_path = os.path.join(search_dir, param_files[timestamp])

    with open(config_path) as f:
        config = json.load(f)

    # Rebuild with the architecture the model was TRAINED with. Without this the
    # smartpool variant is evaluated as a plain-MaxPool network, which silently
    # measures the wrong model.
    variant = resolve_variant(config, search_dir, timestamp)
    switches = VARIANT_SWITCHES[variant]

    model = BayesShipsCNN(
        num_classes=num_classes, device=device,
        activation=config['activation'],
        prior_dist=config['prior'],
        mu=config['prior_params']['mu'],
        b=config['prior_params']['b'],
        **switches,
    ).to(device)

    # Swap in the optimised pooling implementation. SmartPoolFast is verified
    # bit-identical to SmartPool, so this changes runtime only, never results.
    if switches['smartpool_switch'] and fast_smartpool:
        from src.models.components_fast import SmartPoolFast
        ref_pool = model.pool
        model.pool = SmartPoolFast(
            kernel_size=ref_pool.kernel_size, stride=ref_pool.stride,
            threshold=ref_pool.threshold, detect_only=ref_pool.detect_only,
        ).to(device)

    model.load_state_dict(torch.load(model_path, map_location=device))
    return model, param_path, config


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, test_loader = load_data(batch_size=16, fold=args.fold)

    def _assert_fold_matches(config: dict, ts: str) -> None:
        """Refuse to evaluate a model against a test set it may have trained on."""
        expected = args.fold if args.fold is not None else 1
        actual = config.get('fold')
        if actual is None:
            raise ValueError(
                f"config for {ts} has no 'fold' key - run "
                f"scripts/backfill_config_metadata.py first"
            )
        if actual != expected:
            raise ValueError(
                f"fold mismatch for {ts}: model trained on fold {actual}, "
                f"but --fold={expected} selected the fold-{expected} test set. "
                f"This would evaluate on training images."
            )

    prior_map = {'Gaussian_prior': 'gaussian', 'Laplace_prior': 'laplace', 'Uniform_prior': 'uniform'}
    target_prior = prior_map[args.prior]

    all_files = os.listdir(args.search_dir)
    json_files = [f for f in all_files if f.endswith('.json')]
    timestamps = [f[:-5][-16:] for f in json_files]

    excluded = {f[:-4][-16:] for f in os.listdir(args.save_dir) if f.endswith('.csv')}
    timestamps = [ts for ts in timestamps if ts not in excluded]
    print(f"Total timestamps after excluding done: {len(timestamps)}")

    config_files, model_files, param_files = {}, {}, {}
    for ts in timestamps:
        config_files[ts] = next(f for f in all_files if ts in f and f.endswith('.json'))
        model_files[ts] = next(f for f in all_files if ts in f and f.startswith('model'))
        param_files[ts] = next(f for f in all_files if ts in f and f.startswith('param'))

    # filter by prior
    timestamps = [
        ts for ts in timestamps
        if json.load(open(os.path.join(args.search_dir, config_files[ts])))['prior'] == target_prior
    ]
    print(f"After prior filter ({args.prior}): {len(timestamps)}")

    if args.limited_mode:
        timestamps = [
            ts for ts in timestamps
            if json.load(open(os.path.join(args.search_dir, config_files[ts])))['prior_params']['b'] == 1.0
        ]
        print(f"After limited mode (b=1.0) filter: {len(timestamps)}")

    guide_map = {'gaussian': 'AutoNormal', 'laplace': 'AutoLaplace', 'uniform': 'AutoUniform'}
    param_map = {'gaussian': ['locs', 'scales'], 'laplace': ['locs', 'scales'],
                 'uniform': ['lows', 'widths']}
    bit_list = [0, 1, 3, 6, 10, 15, 21]

    for exp_idx, ts in enumerate(timestamps, start=1):
        pyro.clear_param_store()
        send_telegram_message(
            title=f"ShipsNet SEU {exp_idx}/{len(timestamps)}",
            message=f"ts={ts}, prior={args.prior}, limited={args.limited_mode}"
        )

        model, param_store_path, model_config = load_model(
            ts, args.search_dir, config_files, model_files, param_files, device,
            fast_smartpool=args.fast_smartpool,
        )
        prior = model_config['prior']
        _assert_fold_matches(model_config, ts)
        param_unique = guide_map[prior]
        parameter_names = param_map[prior]

        newinj = NewInjector(
            trained_model=model, device=device,
            test_loader=test_loader, num_samples=10,
            pyro_param_store_path=param_store_path,
        )

        results = []
        for attack_loc in ["beginning", "end"]:
            target_idx = 0 if attack_loc == "beginning" else -1
            for layer in ["conv1", "conv2", "fc1"]:
                for module in ["weight", "bias"]:
                    for bit_i in bit_list:
                        for param_name in parameter_names:
                            print(f"SEU: {layer}.{module} bit={bit_i} param={param_name}")
                            result = newinj.run_seu(
                                target_idx, param_unique, param_name,
                                layer, module, bit_i, num_samples=10
                            )
                            results.append({
                                "activation_fn": model_config['activation'],
                                "prior": prior,
                                "fold": model_config.get('fold', 1),
                                "variant": model_config.get('variant', 'base'),
                                "best_accuracy": model_config.get('best_accuracy', None),
                                "prior_mu": model_config['prior_params']['mu'],
                                "prior_b": model_config['prior_params']['b'],
                                "param_type": param_name,
                                "location_index": target_idx,
                                "location_layer": layer,
                                "location_module": module,
                                "bit_index": bit_i,
                                "initial_accuracy": newinj.initial_accuracy,
                                "accuracy_after_seu": newinj.initial_accuracy + result["accuracy_change"],
                                "accuracy_change": result["accuracy_change"],
                                "softmax_difference": result["softmax_difference"],
                                "mean_abs_difference": result["absolute_difference"],
                                "original_bit_condition": result.get("original_bit_condition"),
                                "remarks": result.get("remarks", ""),
                            })

        pd.DataFrame(results).to_csv(
            os.path.join(args.save_dir, f'{ts}.csv'), index=False
        )
        print(f"Results saved for {ts}")

        send_telegram_message(
            title=f"ShipsNet SEU Done {exp_idx}/{len(timestamps)}",
            message=f"ts={ts}, prior={prior}, limited={args.limited_mode}"
        )


if __name__ == "__main__":
    main()
