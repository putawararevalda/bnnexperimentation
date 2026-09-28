"""
Inject Single Event Upsets (SEUs) into trained deterministic CNN models on ShipsNet.

For each .pth checkpoint found in --search-dir, iterates over:
  (layer × module × bit_index) and records accuracy change, softmax difference,
  and absolute weight difference per flip.

Usage examples:
    uv run python scripts/eval_seu_deterministic.py \\
        --search-dir results/shipsnet/deterministic/00 \\
        --save-dir results/shipsnet/seu/deterministic \\
        --model-variant 00

    uv run python scripts/eval_seu_deterministic.py \\
        --search-dir results/shipsnet/deterministic/01 \\
        --save-dir results/shipsnet/seu/deterministic \\
        --model-variant 01
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

from src.data.shipsnet import load_data
from src.evaluation.seu import bitflip_float32_with_original
from src.models.deterministic_cnn import ShipsCNNCustom
from src.utils.notify import send_telegram_message


def parse_args():
    parser = argparse.ArgumentParser(description='SEU evaluation for deterministic ShipsNet models')
    parser.add_argument('--save-dir', type=str, default=None,
                        help='Where to save CSV results. Defaults to results/shipsnet/seu/deterministic/[foldN/]<variant>')
    parser.add_argument('--search-dir', type=str, default=None,
                        help='Directory containing trained .pth checkpoints.')
    parser.add_argument('--fold', type=int, default=None, choices=[1, 2, 3, 4, 5],
                        help='Cross-validation fold (1-5). Omit for legacy split.')
    parser.add_argument('--model-variant', type=str, default='00',
                        choices=['00', '01', '02', '03'],
                        help='Model variant (controls smartpool/dropout). Default: 00')
    return parser.parse_args()


class DeterministicInjector:
    """Injects SEUs directly into deterministic model weights and measures impact."""

    def __init__(self, trained_model, device, test_loader):
        self.device = device
        self.trained_model = trained_model.to(device)
        self.test_loader = test_loader
        self.trained_model.eval()

        initial_labels, initial_preds, _, initial_probs = self._predict_probs()
        self.initial_accuracy = self._accuracy(initial_labels, initial_preds)
        self.initial_probs = np.array(initial_probs)
        print(f"Initial accuracy: {self.initial_accuracy:.4%}")

    def _predict_probs(self):
        all_labels, all_preds, all_logits, all_probs = [], [], [], []
        with torch.no_grad():
            for images, labels in tqdm(self.test_loader, desc="Evaluating", leave=False):
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.trained_model(images)
                preds = outputs.argmax(1)
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
                all_logits.extend(outputs.cpu().numpy())
                all_probs.extend(F.softmax(outputs, dim=1).cpu().numpy())
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
        diffs = np.where(finite, np.max(np.abs(before - safe), axis=1), penalty)
        return float(diffs.mean())

    def run_seu(self, location_index, layer, layer_module, bit_i):
        assert 0 <= bit_i < 32
        target_name = f"{layer}.{layer_module}"

        self.trained_model.eval()
        with torch.no_grad():
            for name, tensor in self.trained_model.named_parameters():
                if name == target_name:
                    break
            else:
                raise ValueError(f"Layer {target_name} not found.")

            flat = tensor.data.clone().view(-1)
            orig_val = flat[location_index].cpu().item()
            flipped, orig_bit = bitflip_float32_with_original(orig_val, bit_i)
            abs_diff = abs(orig_val - flipped)
            tensor.data.view(-1)[location_index] = torch.tensor(
                flipped, dtype=tensor.dtype, device=tensor.device
            )
            print(f"  {target_name}[{location_index}]: {orig_val:.6g} -> {flipped:.6g} (diff {abs_diff:.3g})")

            after_labels, after_preds, after_logits, _ = self._predict_probs()
            accuracy_after = self._accuracy(after_labels, after_preds)
            # `_softmax_diff` applies softmax to its second argument, so that
            # argument MUST be logits. Passing the already-softmaxed `all_probs`
            # here compared p against softmax(p), which adds a near-constant
            # offset (~0.26 for a confident binary model) to every row and made
            # the DNN's softmax_difference column meaningless. The Bayesian
            # scripts pass `after_logits` and were never affected.
            softmax_diff = self._softmax_diff(self.initial_probs, after_logits)

            # restore original value
            tensor.data.view(-1)[location_index] = orig_val

        print(f"  Accuracy after SEU: {accuracy_after:.3%}")
        return {
            "accuracy_change": accuracy_after - self.initial_accuracy,
            "softmax_difference": softmax_diff,
            "absolute_difference": abs_diff,
            "original_bit_condition": orig_bit,
            "remarks": "",
        }


def get_timestamps(directory):
    """Extract 16-char timestamps from .pth filenames."""
    return [f[-20:-4] for f in os.listdir(directory) if f.endswith('.pth')]


def main():
    args = parse_args()
    if args.search_dir:
        search_dir = args.search_dir
    elif args.fold is not None:
        search_dir = f"results/shipsnet/deterministic/fold{args.fold}/{args.model_variant}"
    else:
        search_dir = f"results/shipsnet/deterministic/{args.model_variant}"

    if args.save_dir:
        save_dir = args.save_dir
    elif args.fold is not None:
        save_dir = f"results/shipsnet/seu/deterministic/fold{args.fold}/{args.model_variant}"
    else:
        save_dir = f"results/shipsnet/seu/deterministic/{args.model_variant}"

    os.makedirs(save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, test_loader = load_data(batch_size=16, fold=args.fold, num_workers=0)
    timestamps = get_timestamps(search_dir)
    pth_files = [f for f in os.listdir(search_dir) if f.endswith('.pth')]

    bit_list = [0, 1, 3, 6, 10, 15, 21]

    for exp_idx, ts in enumerate(timestamps, start=1):
        send_telegram_message(
            title=f"DETERMINISTIC SEU {exp_idx}/{len(timestamps)}",
            message=f"ts={ts}, variant={args.model_variant}, save={save_dir}"
        )

        # find the matching .pth file
        model_path = next(
            (os.path.join(search_dir, f) for f in pth_files if ts in f),
            None
        )
        if model_path is None:
            print(f"No .pth found for {ts}, skipping.")
            continue

        # extract activation name from filename: best_model_<act>_<timestamp>.pth
        filename_stem = Path(model_path).stem  # e.g. "best_model_relu_20250101_120000"
        parts = filename_stem.split('_')
        activation = parts[2] if len(parts) >= 3 else 'relu'

        smartpool = args.model_variant == '01'
        dropout = args.model_variant == '02'

        model = ShipsCNNCustom(
            activation=activation,
            smartpool_switch=smartpool,
            dropout_switch=dropout,
        ).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

        injector = DeterministicInjector(trained_model=model, device=device, test_loader=test_loader)

        results = []
        for attack_loc in ["beginning", "end"]:
            target_idx = 0 if attack_loc == "beginning" else -1
            for layer in ["conv1", "conv2", "fc1"]:
                for module in ["weight", "bias"]:
                    for bit_i in bit_list:
                        print(f"SEU: {layer}.{module} bit={bit_i}")
                        result = injector.run_seu(target_idx, layer, module, bit_i)
                        results.append({
                            "fold": args.fold if args.fold is not None else 1,
                            "activation_fn": activation,
                            "model_variant": args.model_variant,
                            "location_index": target_idx,
                            "location_layer": layer,
                            "location_module": module,
                            "bit_index": bit_i,
                            "initial_accuracy": injector.initial_accuracy,
                            "accuracy_after_seu": injector.initial_accuracy + result["accuracy_change"],
                            "accuracy_change": result["accuracy_change"],
                            "softmax_difference": result["softmax_difference"],
                            "mean_abs_difference": result["absolute_difference"],
                            "original_bit_condition": result["original_bit_condition"],
                            "remarks": result["remarks"],
                        })

        pd.DataFrame(results).to_csv(
            os.path.join(save_dir, f'{ts}.csv'), index=False
        )
        print(f"Results saved for {ts}")

        send_telegram_message(
            title=f"DETERMINISTIC SEU Done {exp_idx}/{len(timestamps)}",
            message=f"ts={ts}, variant={args.model_variant}"
        )


if __name__ == "__main__":
    main()
