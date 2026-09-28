"""
Generate datasplit/split_indices_v2.pkl — clean stratified 80/20 split for EuroSAT 10-class.

Replaces the corrupt split_indices.pkl (train was missing SeaLake entirely, had only
88 River images, and shared 3,910 of 5,400 test images with the training set).

Deterministic: seed 42, per-class shuffle, 80% train / 20% test, no overlap.

Usage:
    uv run python scripts/make_eurosat_split_v2.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pickle
from collections import Counter, defaultdict

import numpy as np
import torchvision.datasets as datasets

SEED = 42
SPLIT_RATIO = 0.8
OUT_FILE = "datasplit/split_indices_v2.pkl"


def main():
    dataset = datasets.EuroSAT(root="./data", download=True)
    targets = [dataset.imgs[i][1] for i in range(len(dataset))]

    by_class = defaultdict(list)
    for idx, label in enumerate(targets):
        by_class[label].append(idx)

    rng = np.random.default_rng(SEED)
    train_idx, test_idx = [], []
    for label in sorted(by_class):
        idxs = np.array(by_class[label])
        rng.shuffle(idxs)
        n_train = int(SPLIT_RATIO * len(idxs))
        train_idx.extend(idxs[:n_train].tolist())
        test_idx.extend(idxs[n_train:].tolist())

    # Verification: full coverage, no overlap, all classes present in both splits
    assert len(set(train_idx) & set(test_idx)) == 0, "train/test overlap"
    assert len(train_idx) + len(test_idx) == len(dataset), "coverage mismatch"
    train_counts = Counter(targets[i] for i in train_idx)
    test_counts = Counter(targets[i] for i in test_idx)
    assert set(train_counts) == set(range(10)) and set(test_counts) == set(range(10)), \
        "missing class in a split"

    with open(OUT_FILE, "wb") as f:
        pickle.dump({"train": train_idx, "test": test_idx}, f)

    print(f"Saved {OUT_FILE}: train={len(train_idx)}, test={len(test_idx)}, overlap=0")
    print("train class counts:", dict(sorted(train_counts.items())))
    print("test  class counts:", dict(sorted(test_counts.items())))


if __name__ == "__main__":
    main()
