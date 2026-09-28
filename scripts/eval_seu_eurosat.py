"""
Inject Single Event Upsets (SEUs) into trained Bayesian CNN models on EuroSAT.

For each trained model found in --search-dir, iterates over:
  (layer × module × bit_index × guide_parameter) and records accuracy change,
  softmax difference, and absolute weight difference per flip.

Configs are identified by their full config-file basename, not by the trailing
timestamp, because training can emit two configs sharing one timestamp. Results
whose CSV in --save-dir is already complete (168 rows) are skipped; partial CSVs
are resumed combo-by-combo.

Usage examples:
    uv run python scripts/eval_seu_eurosat.py --prior Gaussian_prior \\
        --search-dir results/eurosat/bayesian \\
        --save-dir results/eurosat/seu

    uv run python scripts/eval_seu_eurosat.py --prior Laplace_prior \\
        --search-dir results/eurosat/bayesian \\
        --save-dir results/eurosat/seu

    uv run python scripts/eval_seu_eurosat.py --prior Uniform_prior \\
        --search-dir results/eurosat/bayesian \\
        --save-dir results/eurosat/seu
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import copy
import csv
import json
import math
import os

import numpy as np
import pandas as pd
import pyro
import torch
import torch.nn.functional as F
from pyro.infer.autoguide import AutoNormal
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

from src.data.eurosat import load_data
from src.evaluation.seu import bitflip_float32_with_original
from src.models.bayesian_cnn import BayesShipsCNN
from src.utils.guide import AutoLaplace, AutoUniform
from src.utils.notify import send_telegram_message


# EuroSAT variant dirs -> model architecture switches. SmartPool is ACTIVE at
# eval time, so the smartpool variant must be reconstructed with it enabled or
# SEU injects into the wrong (plain MaxPool) forward pass.
DIR_TO_VARIANT = {
    "results_eurosat_v02_00": "base",
    "results_eurosat_v02_01": "smartpool",
    "results_eurosat_v02_02": "dropout",
    "results_eurosat_v02_03": "weight_decay",
}


def variant_from_search_dir(search_dir: str) -> str:
    """Infer the model variant from the search-dir name; 'base' if unrecognized."""
    return DIR_TO_VARIANT.get(os.path.basename(os.path.normpath(search_dir)), "base")


def parse_args():
    parser = argparse.ArgumentParser(description='SEU evaluation for Bayesian EuroSAT models')
    parser.add_argument('--prior', type=str, default='Gaussian_prior',
                        choices=['Gaussian_prior', 'Laplace_prior', 'Uniform_prior'])
    parser.add_argument('--save-dir', type=str, default='results/eurosat/seu',
                        help='Where to save CSV results.')
    parser.add_argument('--search-dir', type=str, default='results/eurosat/bayesian',
                        help='Directory containing trained model artifacts.')
    parser.add_argument('--limited-mode', dest='limited_mode', action='store_true',
                        help='Only evaluate models with prior scale b=1.0')
    parser.add_argument('--no-limited-mode', dest='limited_mode', action='store_false')
    parser.add_argument('--fast-smartpool', action='store_true',
                        help='Use SmartPoolFast (bit-identical to SmartPool, ~15x '
                             'faster model forward). Only affects smartpool variants.')
    parser.set_defaults(limited_mode=False)
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

        # Loaded once and reused via deepcopy in run_seu() instead of re-reading
        # this file from disk on every one of the ~168 flips per config.
        self._clean_state = torch.load(pyro_param_store_path, weights_only=False)

        self._build_guide()
        pyro.clear_param_store()
        pyro.get_param_store().set_state(copy.deepcopy(self._clean_state))

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
            raise ValueError(f"Unsupported prior for EuroSAT SEU eval: {prior}")

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
        pyro.get_param_store().set_state(copy.deepcopy(self._clean_state))

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


CSV_FIELDS = [
    "activation_fn", "prior", "variant", "best_accuracy", "prior_mu", "prior_b",
    "param_type", "location_index", "location_layer", "location_module", "bit_index",
    "initial_accuracy", "accuracy_after_seu", "accuracy_change", "softmax_difference",
    "mean_abs_difference", "original_bit_condition", "remarks",
]
TOTAL_COMBOS_PER_CONFIG = 2 * 3 * 2 * 7 * 2  # attack_loc x layer x module x bit x param_name


def load_completed_combos(csv_path):
    """Reads a possibly-partial result CSV (e.g. left behind by a killed/crashed
    run) and returns the set of (location_index, layer, module, bit_index,
    param_type) combos already recorded, so the sweep can skip them on resume.
    Tolerates a truncated last line if the process died mid-write."""
    if not os.path.exists(csv_path):
        return set()
    try:
        df = pd.read_csv(csv_path)
    except pd.errors.ParserError:
        # Last line may be a partial write from a hard kill; drop it and retry.
        with open(csv_path, "r") as f:
            lines = f.readlines()
        with open(csv_path, "w") as f:
            f.writelines(lines[:-1])
        df = pd.read_csv(csv_path)
    combos = set()
    for _, row in df.iterrows():
        combos.add((
            int(row["location_index"]), row["location_layer"],
            row["location_module"], int(row["bit_index"]), row["param_type"],
        ))
    return combos


def _csv_belongs_to(csv_path, config):
    """True if an existing result CSV was produced by `config`.

    Used only to decide who keeps a legacy `{timestamp}.csv` when two configs
    share a timestamp. Matches on the columns that identify a config; a
    header-only or unreadable file counts as 'not mine' so the caller falls
    back to a unique filename rather than appending to someone else's results.
    """
    try:
        df = pd.read_csv(csv_path, nrows=1)
    except (pd.errors.ParserError, pd.errors.EmptyDataError, OSError):
        return False
    if df.empty:
        return False
    row = df.iloc[0]
    # prior_b is compared with a tolerance: the CSV round-trip drops the last
    # digit of the float repr (0.10000000149011612 -> 0.1000000014901161).
    # The b grid is {10.0, 1.0, 0.1}, so any loose tolerance separates them.
    return (
        row["activation_fn"] == config["activation"]
        and row["prior"] == config["prior"]
        and math.isclose(float(row["prior_b"]), float(config["prior_params"]["b"]),
                         rel_tol=1e-6)
    )


def discover_configs(search_dir, save_dir):
    """Enumerate trained configs in `search_dir`, keyed by full config basename.

    Identity is the config file's stem (e.g. `_tanh_laplace_20260711_053841`),
    not the trailing timestamp. Training can emit two configs with the same
    timestamp -- EuroSAT v02_02 has two such collisions -- and keying on the
    timestamp alone silently made one config of each pair unreachable.

    Each entry is a dict with: stem, ts, tag, config_file, model_file,
    param_file, config, csv_path.

    `csv_path` stays `{ts}.csv` (the historical name) for every unambiguous
    timestamp, so all previously collected results remain resumable. Only when
    a timestamp is shared does a config fall back to `{stem}.csv` -- and even
    then, whichever config already owns the legacy file keeps it.
    """
    all_files = os.listdir(search_dir)
    json_files = sorted(f for f in all_files if f.endswith('.json') and f.startswith('config'))

    ts_counts = {}
    for f in json_files:
        ts_counts[f[:-5][-16:]] = ts_counts.get(f[:-5][-16:], 0) + 1

    entries = []
    for cfg_file in json_files:
        stem = cfg_file[len('config'):-len('.json')]
        ts, tag = stem[-16:], stem[:-16]

        def _find(prefix, suffix):
            match = [f for f in all_files
                     if f.startswith(prefix + tag + '_') and f.endswith(ts + suffix)]
            if not match:
                raise FileNotFoundError(
                    f"no {prefix}* artifact for config '{cfg_file}' in {search_dir}")
            return sorted(match)[0]

        with open(os.path.join(search_dir, cfg_file)) as f:
            config = json.load(f)

        csv_path = os.path.join(save_dir, f'{ts}.csv')
        if ts_counts[ts] > 1 and not (
                os.path.exists(csv_path) and _csv_belongs_to(csv_path, config)):
            csv_path = os.path.join(save_dir, f'{stem}.csv')

        entries.append({
            'stem': stem, 'ts': ts, 'tag': tag,
            'config_file': cfg_file,
            'model_file': _find('model', '.pth'),
            'param_file': _find('param_store', '.pkl'),
            'config': config,
            'csv_path': csv_path,
        })
    return entries


def load_model(timestamp, search_dir, config_files, model_files, param_files,
               device, num_classes=10, smartpool_switch=False, dropout_switch=False,
               fast_smartpool=False):
    config_path = os.path.join(search_dir, config_files[timestamp])
    model_path = os.path.join(search_dir, model_files[timestamp])
    param_path = os.path.join(search_dir, param_files[timestamp])

    with open(config_path) as f:
        config = json.load(f)

    model = BayesShipsCNN(
        num_classes=num_classes, device=device,
        activation=config['activation'],
        prior_dist=config['prior'],
        mu=config['prior_params']['mu'],
        b=config['prior_params']['b'],
        smartpool_switch=smartpool_switch,
        dropout_switch=dropout_switch,
    ).to(device)

    # Swap in the optimised pooling implementation. SmartPoolFast is verified
    # bit-identical to SmartPool, so this changes runtime only, never results.
    if smartpool_switch and fast_smartpool:
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
    _, test_loader = load_data(batch_size=54)

    variant = variant_from_search_dir(args.search_dir)
    smartpool_switch = variant == "smartpool"
    dropout_switch = variant == "dropout"
    print(f"Search-dir variant: {variant} "
          f"(smartpool={smartpool_switch}, dropout={dropout_switch})")

    prior_map = {'Gaussian_prior': 'gaussian', 'Laplace_prior': 'laplace',
                 'Uniform_prior': 'uniform'}
    target_prior = prior_map[args.prior]

    entries = discover_configs(args.search_dir, args.save_dir)
    print(f"Configs found: {len(entries)}")

    entries = [e for e in entries if e['config']['prior'] == target_prior]
    print(f"After prior filter ({args.prior}): {len(entries)}")

    if args.limited_mode:
        entries = [e for e in entries if e['config']['prior_params']['b'] == 1.0]
        print(f"After limited mode (b=1.0) filter: {len(entries)}")

    # Only skip a config entirely if its own CSV already has all 168 rows -- a
    # partial CSV (from a killed/crashed run) is resumed inside the loop below
    # instead of being treated as done.
    entries = [
        e for e in entries
        if len(load_completed_combos(e['csv_path'])) < TOTAL_COMBOS_PER_CONFIG
    ]
    print(f"After excluding completed: {len(entries)}")

    config_files = {e['stem']: e['config_file'] for e in entries}
    model_files = {e['stem']: e['model_file'] for e in entries}
    param_files = {e['stem']: e['param_file'] for e in entries}

    guide_map = {'gaussian': 'AutoNormal', 'laplace': 'AutoLaplace', 'uniform': 'AutoUniform'}
    param_map = {'gaussian': ['locs', 'scales'], 'laplace': ['locs', 'scales'],
                 'uniform': ['lows', 'widths']}
    bit_list = [0, 1, 3, 6, 10, 15, 21]

    for exp_idx, entry in enumerate(entries, start=1):
        stem, ts = entry['stem'], entry['ts']
        pyro.clear_param_store()
        send_telegram_message(
            title=f"EuroSAT SEU {exp_idx}/{len(entries)}",
            message=f"config={stem}, prior={args.prior}, limited={args.limited_mode}"
        )

        model, param_store_path, model_config = load_model(
            stem, args.search_dir, config_files, model_files, param_files, device,
            smartpool_switch=smartpool_switch, dropout_switch=dropout_switch,
            fast_smartpool=args.fast_smartpool,
        )
        prior = model_config['prior']
        param_unique = guide_map[prior]
        parameter_names = param_map[prior]

        newinj = NewInjector(
            trained_model=model, device=device,
            test_loader=test_loader, num_samples=10,
            pyro_param_store_path=param_store_path,
        )

        csv_path = entry['csv_path']
        done_combos = load_completed_combos(csv_path)
        if done_combos:
            print(f"Resuming {stem}: {len(done_combos)}/{TOTAL_COMBOS_PER_CONFIG} combos already done")

        file_exists = os.path.exists(csv_path)
        with open(csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            if not file_exists:
                writer.writeheader()
                f.flush()

            for attack_loc in ["beginning", "end"]:
                target_idx = 0 if attack_loc == "beginning" else -1
                for layer in ["conv1", "conv2", "fc1"]:
                    for module in ["weight", "bias"]:
                        for bit_i in bit_list:
                            for param_name in parameter_names:
                                combo_key = (target_idx, layer, module, bit_i, param_name)
                                if combo_key in done_combos:
                                    continue
                                print(f"SEU: {layer}.{module} bit={bit_i} param={param_name}")
                                result = newinj.run_seu(
                                    target_idx, param_unique, param_name,
                                    layer, module, bit_i, num_samples=10
                                )
                                row = {
                                    "activation_fn": model_config['activation'],
                                    "prior": prior,
                                    "variant": variant,
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
                                }
                                writer.writerow(row)
                                f.flush()

        print(f"Results saved for {stem} -> {os.path.basename(csv_path)}")

        send_telegram_message(
            title=f"EuroSAT SEU Done {exp_idx}/{len(entries)}",
            message=f"config={stem}, prior={prior}, limited={args.limited_mode}"
        )


if __name__ == "__main__":
    main()
