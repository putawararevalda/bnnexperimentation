"""
SEU Robustness Evaluation for Deterministic ShipsNet Baselines on Folds 2–5.

Evaluates trained CNN checkpoints across:
  - Folds: 2, 3, 4, 5 (or user-specified subset)
  - Variants: 00 (base), 01 (smartpool), 02 (dropout), 03 (weight decay)
  - Activations: 7 canonical activations (relu, tanh, sigmoid, sin, relu6, actWG, actRWG)
  - 84 SEU bitflip combinations per model:
      (2 locations [beginning, end]) x (3 layers [conv1, conv2, fc1]) x (2 modules [weight, bias]) x (7 bits [0, 1, 3, 6, 10, 15, 21])

Outputs:
  Saved to: results/shipsnet/seu/deterministic/fold{fold}/{variant}/_{timestamp}.csv
  (Matches the exact 84-row format of Fold 1)

Usage examples:
    # Run full sweep across folds 2-5 (all variants):
    uv run python scripts/eval_seu_deterministic_folds2to5.py

    # Dry-run to verify discovered models and completion status:
    uv run python scripts/eval_seu_deterministic_folds2to5.py --dry-run

    # Run fold 2 only:
    uv run python scripts/eval_seu_deterministic_folds2to5.py --folds 2

    # Run specific variant on fold 3:
    uv run python scripts/eval_seu_deterministic_folds2to5.py --folds 3 --variants 00
"""

import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import csv
import os
import re
import time

SUPPORTED_ACTIVATIONS = ["relu", "tanh", "sigmoid", "sin", "relu6", "actWG", "actRWG"]
BIT_LIST = [0, 1, 3, 6, 10, 15, 21]
ATTACK_LOCATIONS = [("beginning", 0), ("end", -1)]
TARGET_LAYERS = ["conv1", "conv2", "fc1"]
TARGET_MODULES = ["weight", "bias"]



def parse_args():
    parser = argparse.ArgumentParser(
        description="SEU robustness evaluation for deterministic CNNs on ShipsNet folds 2-5"
    )
    parser.add_argument(
        "--folds",
        type=int,
        nargs="+",
        default=[2, 3, 4, 5],
        choices=[1, 2, 3, 4, 5],
        help="Folds to evaluate. Default: 2 3 4 5",
    )
    parser.add_argument(
        "--variants",
        type=str,
        nargs="+",
        default=["00", "01", "02", "03"],
        choices=["00", "01", "02", "03"],
        help="Model variants to evaluate. Default: 00 01 02 03",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for test evaluation. Default: 16",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use ('cuda' or 'cpu'). Default: auto-detect",
    )
    parser.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        help="Do not skip already completed CSV files (re-evaluates everything).",
    )
    parser.set_defaults(resume=True)
    parser.add_argument(
        "--all-checkpoints",
        action="store_true",
        help="Evaluate all checkpoints instead of selecting the best one per activation.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Scan and list all model checkpoints and target CSV paths without running SEUs.",
    )
    return parser.parse_args()


def extract_activation(filename: str) -> str:
    """Robustly extract the canonical activation name from a checkpoint filename."""
    fn = filename.lower()
    if "actrwg" in fn:
        return "actRWG"
    elif "actwg" in fn:
        return "actWG"
    elif "relu6" in fn:
        return "relu6"
    elif "relu" in fn:
        return "relu"
    elif "sigmoid" in fn:
        return "sigmoid"
    elif "sin" in fn:
        return "sin"
    elif "tanh" in fn:
        return "tanh"
    raise ValueError(f"Could not determine activation function from filename: '{filename}'")


def extract_timestamp(filename: str) -> str:
    """Extract YYYYMMDD_HHMMSS timestamp from filename."""
    m = re.search(r"(\d{8}_\d{6})", filename)
    if not m:
        raise ValueError(f"Could not extract timestamp from filename: '{filename}'")
    return m.group(1)


def is_csv_complete(csv_path: Path) -> bool:
    """Check if the SEU output CSV already exists and has exactly 85 rows (1 header + 84 data)."""
    if not csv_path.exists():
        return False
    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            line_count = sum(1 for _ in f)
        return line_count >= 85
    except Exception:
        return False


def get_checkpoint_val_acc(pth_file: Path) -> float:
    """Read the best validation accuracy from the corresponding training log CSV."""
    ts = extract_timestamp(pth_file.name)
    parent_dir = pth_file.parent
    for log_path in parent_dir.glob(f"*{ts}*.csv"):
        try:
            with open(log_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                val_accs = [
                    float(r["val_acc"])
                    for r in reader
                    if "val_acc" in r and r["val_acc"] != ""
                ]
                if val_accs:
                    return max(val_accs)
        except Exception:
            continue
    return 0.0


def select_target_checkpoints(directory: Path, select_best: bool = True):
    """
    Return a list of (pth_path, activation, timestamp) tuples.
    If select_best is True, picks only the single best checkpoint per activation.
    """
    pth_files = sorted(directory.glob("*.pth"))
    if not pth_files:
        return []

    if not select_best:
        return [(p, extract_activation(p.name), extract_timestamp(p.name)) for p in pth_files]

    # Group by activation and choose the one with the highest validation accuracy (or latest timestamp)
    by_act = {}
    for p in pth_files:
        try:
            act = extract_activation(p.name)
            ts = extract_timestamp(p.name)
        except Exception as e:
            print(f"  [Warning] Skipping unrecognized file {p.name}: {e}")
            continue

        val_acc = get_checkpoint_val_acc(p)
        if act not in by_act:
            by_act[act] = (p, act, ts, val_acc)
        else:
            prev_p, prev_act, prev_ts, prev_acc = by_act[act]
            if val_acc > prev_acc or (val_acc == prev_acc and ts > prev_ts):
                by_act[act] = (p, act, ts, val_acc)

    return [(t[0], t[1], t[2]) for t in by_act.values()]


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

    def _predict_probs(self):
        all_labels, all_preds, all_logits, all_probs = [], [], [], []
        with torch.no_grad():
            for images, labels in self.test_loader:
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
        return float(np.trace(cm) / np.sum(cm))

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
                raise ValueError(f"Layer {target_name} not found in model.")

            flat = tensor.data.clone().view(-1)
            orig_val = flat[location_index].cpu().item()
            flipped, orig_bit = bitflip_float32_with_original(orig_val, bit_i)
            abs_diff = abs(orig_val - flipped)

            tensor.data.view(-1)[location_index] = torch.tensor(
                flipped, dtype=tensor.dtype, device=tensor.device
            )

            after_labels, after_preds, after_logits, _ = self._predict_probs()
            accuracy_after = self._accuracy(after_labels, after_preds)

            # Clean single-softmax difference: passing after_logits avoids double-softmax bug
            softmax_diff = self._softmax_diff(self.initial_probs, after_logits)

            # Restore original parameter value
            tensor.data.view(-1)[location_index] = orig_val

        return {
            "accuracy_change": accuracy_after - self.initial_accuracy,
            "softmax_difference": softmax_diff,
            "absolute_difference": abs_diff,
            "original_bit_condition": orig_bit,
            "remarks": "",
        }


def main():
    args = parse_args()
    print(f"=== ShipsNet Deterministic SEU Evaluation (Folds 2–5) ===")
    print(f"Target Folds: {args.folds}")
    print(f"Target Variants: {args.variants}")
    print(f"Resume existing: {args.resume}")
    print(f"Select best checkpoint only: {not args.all_checkpoints}")
    print(f"Dry-run: {args.dry_run}\n")

    root_dir = Path("results/shipsnet")
    det_search_root = root_dir / "deterministic"
    seu_save_root = root_dir / "seu/deterministic"

    # 1. Discover all checkpoints to evaluate
    tasks = []
    for fold in args.folds:
        for variant in args.variants:
            search_dir = det_search_root / f"fold{fold}" / variant
            save_dir = seu_save_root / f"fold{fold}" / variant

            if not search_dir.exists():
                print(f"[Warning] Directory {search_dir} does not exist. Skipping.")
                continue

            checkpoints = select_target_checkpoints(
                search_dir, select_best=not args.all_checkpoints
            )
            for pth_path, act, ts in checkpoints:
                out_csv = save_dir / f"_{ts}.csv"
                tasks.append(
                    {
                        "fold": fold,
                        "variant": variant,
                        "activation": act,
                        "timestamp": ts,
                        "pth_path": pth_path,
                        "out_csv": out_csv,
                        "save_dir": save_dir,
                    }
                )

    print(f"Found total of {len(tasks)} model checkpoints across specified folds.")

    if args.dry_run:
        print("\n--- Dry Run Inventory ---")
        for i, t in enumerate(tasks, 1):
            status = "DONE" if is_csv_complete(t["out_csv"]) else "PENDING"
            print(
                f"[{i:02d}/{len(tasks):02d}] Fold {t['fold']} | Var {t['variant']} | Act: {t['activation']:<8} "
                f"| Status: {status:<7} | CSV: {t['out_csv']}"
            )
        pending_count = sum(1 for t in tasks if not is_csv_complete(t["out_csv"]))
        print(f"\nDry run complete. {pending_count} pending / {len(tasks)} total.")
        return

    # 2. Execute SEU evaluations fold by fold to avoid reloading dataset
    global torch, F, confusion_matrix, bitflip_float32_with_original, np
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

    device = torch.device(
        args.device
        if args.device
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Execution Device: {device}")

    completed_runs = 0
    skipped_runs = 0
    start_total_time = time.time()

    for fold in args.folds:
        fold_tasks = [t for t in tasks if t["fold"] == fold]
        if not fold_tasks:
            continue

        print(f"\n=======================================================")
        print(f"  Loading Test Dataset for Fold {fold}...")
        print(f"=======================================================")
        _, test_loader = load_data(
            batch_size=args.batch_size, fold=fold, num_workers=0
        )

        for t in fold_tasks:
            out_csv = t["out_csv"]
            if args.resume and is_csv_complete(out_csv):
                print(f"  [Skip] Fold {fold} Var {t['variant']} {t['activation']} already complete: {out_csv.name}")
                skipped_runs += 1
                continue

            t["save_dir"].mkdir(parents=True, exist_ok=True)
            act = t["activation"]
            variant = t["variant"]
            smartpool = (variant == "01")
            dropout = (variant == "02")

            print(f"\nEvaluating: Fold {fold} | Variant {variant} | Activation {act} | Timestamp {t['timestamp']}")
            print(f"  Checkpoint: {t['pth_path'].name}")

            # Instantiate and load model weights
            model = ShipsCNNCustom(
                activation=act,
                smartpool_switch=smartpool,
                dropout_switch=dropout,
            ).to(device)
            model.load_state_dict(torch.load(t["pth_path"], map_location=device))
            model.eval()

            # Initialize injector and measure pre-SEU baseline
            injector = DeterministicInjector(
                trained_model=model, device=device, test_loader=test_loader
            )
            print(f"  Initial test accuracy: {injector.initial_accuracy:.4%}")

            # Run 84-injection SEU sweep
            results = []
            progress_bar = tqdm(
                total=len(ATTACK_LOCATIONS) * len(TARGET_LAYERS) * len(TARGET_MODULES) * len(BIT_LIST),
                desc="  Injecting SEUs",
                leave=False,
            )

            for attack_loc_name, target_idx in ATTACK_LOCATIONS:
                for layer in TARGET_LAYERS:
                    for module in TARGET_MODULES:
                        for bit_i in BIT_LIST:
                            res = injector.run_seu(target_idx, layer, module, bit_i)
                            results.append(
                                {
                                    "fold": fold,
                                    "activation_fn": act,
                                    "model_variant": variant,
                                    "location_index": target_idx,
                                    "location_layer": layer,
                                    "location_module": module,
                                    "bit_index": bit_i,
                                    "initial_accuracy": injector.initial_accuracy,
                                    "accuracy_after_seu": injector.initial_accuracy + res["accuracy_change"],
                                    "accuracy_change": res["accuracy_change"],
                                    "softmax_difference": res["softmax_difference"],
                                    "mean_abs_difference": res["absolute_difference"],
                                    "original_bit_condition": res["original_bit_condition"],
                                    "remarks": res["remarks"],
                                }
                            )
                            progress_bar.update(1)

            progress_bar.close()

            # Save to CSV
            df = pd.DataFrame(results)
            df.to_csv(out_csv, index=False)
            print(f"  ↳ Saved 84 rows to: {out_csv}")
            completed_runs += 1

            send_telegram_message(
                title=f"DETERMINISTIC SEU DONE [Fold {fold}]",
                message=f"Fold {fold}, Variant {variant}, Act {act}, InitAcc {injector.initial_accuracy:.2%}",
            )

    elapsed = time.time() - start_total_time
    print(f"\n=======================================================")
    print(f"All deterministic evaluations finished in {elapsed/60:.1f} minutes.")
    print(f"Total processed: {completed_runs} | Skipped: {skipped_runs}")
    print(f"=======================================================")


if __name__ == "__main__":
    main()
