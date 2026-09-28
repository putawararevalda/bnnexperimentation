"""Generate the ShipsNet k=5 stratified fold manifest.

Fold 1 is pinned to the existing 80/20 split so all previously trained models
remain valid as fold-1 results.

Usage:
    uv run --link-mode=copy python scripts/make_shipsnet_folds.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import logging
import pickle
import time

from torchvision.datasets import ImageFolder

from src.data.folds import FOLD_FILE, build_folds, validate_folds

DATA_ROOT = "data/shipsnet/foldered"
LEGACY_SPLIT = "datasplit/shipsnet_split_indices.pkl"

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate ShipsNet k-fold manifest")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default=FOLD_FILE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    dataset = ImageFolder(root=DATA_ROOT)
    targets = list(dataset.targets)
    logger.info("Dataset: %d images, classes=%s", len(targets), dataset.classes)

    with open(LEGACY_SPLIT, "rb") as f:
        legacy = pickle.load(f)
    legacy_test = list(legacy["test"])
    logger.info("Legacy split: %d train / %d test", len(legacy["train"]), len(legacy_test))

    folds = build_folds(targets, legacy_test, k=args.k, seed=args.seed)
    validate_folds(folds, targets, legacy_test)

    if set(folds[0]["train"]) != set(legacy["train"]):
        raise ValueError("fold 1 train set does not match the legacy split")
    logger.info("Fold 1 matches the legacy split exactly (train and test)")

    manifest = {
        "k": args.k,
        "folds": folds,
        "meta": {
            "seed": args.seed,
            "source_split": LEGACY_SPLIT,
            "created": time.strftime("%Y-%m-%d"),
            "classes": dataset.classes,
            "class_counts": {c: targets.count(i) for i, c in enumerate(dataset.classes)},
        },
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "wb") as f:
        pickle.dump(manifest, f)
    logger.info("Wrote %s", args.out)

    for i, fold in enumerate(folds, start=1):
        counts = {c: 0 for c in dataset.classes}
        for idx in fold["test"]:
            counts[dataset.classes[targets[idx]]] += 1
        logger.info("  fold %d: train=%d test=%d %s",
                    i, len(fold["train"]), len(fold["test"]), counts)


if __name__ == "__main__":
    main()
