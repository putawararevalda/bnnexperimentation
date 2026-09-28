# ShipsNet k=5 Cross-Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add k=5 stratified cross-validation to the ShipsNet BNN training and SEU pipeline so every reported metric carries mean ± standard deviation across folds.

**Architecture:** A pure fold-partitioning module (`src/data/folds.py`) generates a manifest pickle in which **fold 1 is pinned to the existing 80/20 split**, making the 252 already-trained models reusable as fold 1. A `fold` parameter is threaded through the data loader, training script, and SEU script; a new aggregation script reduces the resulting CSVs across folds into paper-ready tables.

**Tech Stack:** Python 3.11+, PyTorch, Pyro, NumPy, pandas, scikit-learn, pytest, `uv` for environment management.

**Spec:** `docs/superpowers/specs/2026-07-22-shipsnet-kfold-cv-design.md`

## Global Constraints

- Python `>=3.11` (per `pyproject.toml`). Type hints on all new functions.
- All Python commands run through `uv run`. Never `pip`, `conda`, or bare `python`.
- `torch.manual_seed(42)` stays **fixed across all folds**. Only the split varies.
- Fold partitioning seed is `42`, via `numpy.random.default_rng(42)`.
- `fold=None` must preserve the existing code path exactly. No existing script may change behaviour when the new flag is absent.
- Dataset is 4000 ShipsNet images: 3000 `no_ship` (label 0) + 1000 `ship` (label 1). k=5 → 800 per fold → exactly 600 label-0 + 200 label-1 per fold.
- Fold indices in the manifest are **0-based** (`folds[0]` is fold 1); all CLI arguments and recorded metadata are **1-based** (`--fold 1`..`--fold 5`).
- Variant vocabulary is exactly: `base`, `smartpool`, `dropout`, `weight_decay`.
- Directory → variant map: `results_shipsnet_v02_00`→`base`, `v02_01`→`smartpool`, `v02_02`→`dropout`, `v02_03`→`weight_decay`.
- Never move or rewrite existing fold-1 model artifacts. Backfill only adds JSON keys.
- Logging via `logging`, not `print()`, in new `src/` modules. Scripts may print user-facing progress.

---

### Task 1: Fold partitioning module

The heart of the project. Pure functions over index lists — no file or dataset I/O — so correctness is testable with synthetic data in milliseconds instead of by loading 4000 images.

**Files:**
- Create: `src/data/folds.py`
- Create: `tests/test_folds.py`

**Interfaces:**
- Consumes: nothing (leaf module)
- Produces:
  - `FOLD_FILE: str = "datasplit/shipsnet_folds_k5.pkl"`
  - `build_folds(targets: Sequence[int], legacy_test: Sequence[int], k: int = 5, seed: int = 42) -> list[dict[str, list[int]]]` — returns `k` dicts, each `{"train": [...], "test": [...]}`, index 0 = fold 1
  - `validate_folds(folds: Sequence[Mapping[str, Sequence[int]]], targets: Sequence[int], legacy_test: Sequence[int]) -> None` — raises `ValueError` on any violation
  - `load_folds(path: str = FOLD_FILE) -> dict` — returns `{"k": int, "folds": [...], "meta": {...}}`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_folds.py`:

```python
"""Unit tests for stratified fold partitioning."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from src.data.folds import build_folds, validate_folds


def make_targets(n_zero: int = 3000, n_one: int = 1000) -> list[int]:
    """ImageFolder orders classes alphabetically: no_ship(0) then ship(1)."""
    return [0] * n_zero + [1] * n_one


def make_legacy_test(targets: list[int], n_zero: int = 600, n_one: int = 200) -> list[int]:
    zeros = [i for i, t in enumerate(targets) if t == 0][:n_zero]
    ones = [i for i, t in enumerate(targets) if t == 1][:n_one]
    return sorted(zeros + ones)


def test_produces_k_folds():
    targets = make_targets()
    folds = build_folds(targets, make_legacy_test(targets), k=5, seed=42)
    assert len(folds) == 5


def test_fold_one_is_legacy_split():
    targets = make_targets()
    legacy_test = make_legacy_test(targets)
    folds = build_folds(targets, legacy_test, k=5, seed=42)
    assert set(folds[0]["test"]) == set(legacy_test)
    expected_train = set(range(len(targets))) - set(legacy_test)
    assert set(folds[0]["train"]) == expected_train


def test_test_blocks_are_disjoint_and_exhaustive():
    targets = make_targets()
    folds = build_folds(targets, make_legacy_test(targets), k=5, seed=42)
    blocks = [set(f["test"]) for f in folds]
    union = set()
    for b in blocks:
        assert not (union & b), "test blocks overlap"
        union |= b
    assert union == set(range(len(targets)))


def test_every_block_is_stratified():
    targets = make_targets()
    folds = build_folds(targets, make_legacy_test(targets), k=5, seed=42)
    for i, f in enumerate(folds):
        counts = {0: 0, 1: 0}
        for idx in f["test"]:
            counts[targets[idx]] += 1
        assert counts == {0: 600, 1: 200}, f"fold {i + 1} not stratified: {counts}"


def test_train_and_test_are_complementary():
    targets = make_targets()
    folds = build_folds(targets, make_legacy_test(targets), k=5, seed=42)
    for f in folds:
        assert set(f["train"]) & set(f["test"]) == set()
        assert len(f["train"]) == 3200
        assert len(f["test"]) == 800


def test_is_deterministic_for_same_seed():
    targets = make_targets()
    legacy = make_legacy_test(targets)
    a = build_folds(targets, legacy, k=5, seed=42)
    b = build_folds(targets, legacy, k=5, seed=42)
    assert a == b


def test_different_seed_changes_folds_2_to_5_only():
    targets = make_targets()
    legacy = make_legacy_test(targets)
    a = build_folds(targets, legacy, k=5, seed=42)
    b = build_folds(targets, legacy, k=5, seed=7)
    assert a[0] == b[0], "fold 1 must be pinned to the legacy split regardless of seed"
    assert a[1:] != b[1:]


def test_rejects_non_divisible_legacy_test():
    targets = make_targets()
    bad_legacy = make_legacy_test(targets)[:799]
    with pytest.raises(ValueError, match="not divisible"):
        build_folds(targets, bad_legacy, k=5, seed=42)


def test_validate_accepts_good_folds():
    targets = make_targets()
    legacy = make_legacy_test(targets)
    folds = build_folds(targets, legacy, k=5, seed=42)
    validate_folds(folds, targets, legacy)


def test_validate_rejects_overlapping_folds():
    targets = make_targets()
    legacy = make_legacy_test(targets)
    folds = build_folds(targets, legacy, k=5, seed=42)
    folds[1]["test"][0] = folds[0]["test"][0]
    with pytest.raises(ValueError):
        validate_folds(folds, targets, legacy)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run --with pytest,numpy pytest tests/test_folds.py -v
```

Expected: collection error — `ModuleNotFoundError: No module named 'src.data.folds'`

- [ ] **Step 3: Implement the module**

Create `src/data/folds.py`:

```python
"""Stratified k-fold partitioning for ShipsNet, pinned to the existing 80/20 split.

Fold 1's test block is the legacy split's test set verbatim, which makes every
already-trained model a valid fold-1 result. See
docs/superpowers/specs/2026-07-22-shipsnet-kfold-cv-design.md
"""
import logging
import pickle
from collections import defaultdict
from typing import Mapping, Sequence

import numpy as np

logger = logging.getLogger(__name__)

FOLD_FILE = "datasplit/shipsnet_folds_k5.pkl"


def build_folds(
    targets: Sequence[int],
    legacy_test: Sequence[int],
    k: int = 5,
    seed: int = 42,
) -> list[dict[str, list[int]]]:
    """Partition indices into k stratified folds, with fold 1 pinned to legacy_test.

    Args:
        targets: Class label per dataset index (ImageFolder ordering).
        legacy_test: Test indices of the existing split; becomes fold 1's test block.
        k: Number of folds.
        seed: Seed for shuffling the remaining indices before chunking.

    Returns:
        k dicts of {"train": [...], "test": [...]}, index 0 being fold 1.

    Raises:
        ValueError: If the dataset does not partition evenly into k stratified blocks.
    """
    n = len(targets)
    all_idx = set(range(n))
    legacy_test_set = set(legacy_test)

    if not legacy_test_set <= all_idx:
        raise ValueError("legacy_test contains indices outside the dataset")
    if n % k != 0:
        raise ValueError(f"dataset size {n} not divisible by k={k}")
    block_size = n // k
    if len(legacy_test_set) != block_size:
        raise ValueError(
            f"legacy test set has {len(legacy_test_set)} indices, "
            f"not divisible into k={k} blocks of {block_size}"
        )

    remaining = sorted(all_idx - legacy_test_set)
    by_class: dict[int, list[int]] = defaultdict(list)
    for idx in remaining:
        by_class[targets[idx]].append(idx)

    rng = np.random.default_rng(seed)
    blocks: list[list[int]] = [[] for _ in range(k - 1)]
    for label, idxs in sorted(by_class.items()):
        if len(idxs) % (k - 1) != 0:
            raise ValueError(
                f"class {label} has {len(idxs)} remaining indices, "
                f"not divisible into {k - 1} blocks"
            )
        shuffled = np.array(idxs)
        rng.shuffle(shuffled)
        for j, chunk in enumerate(np.array_split(shuffled, k - 1)):
            blocks[j].extend(int(i) for i in chunk)

    test_blocks = [sorted(legacy_test_set)] + [sorted(b) for b in blocks]
    folds: list[dict[str, list[int]]] = []
    for block in test_blocks:
        folds.append({
            "train": sorted(all_idx - set(block)),
            "test": block,
        })
    return folds


def validate_folds(
    folds: Sequence[Mapping[str, Sequence[int]]],
    targets: Sequence[int],
    legacy_test: Sequence[int],
) -> None:
    """Assert every invariant the downstream pipeline depends on.

    Raises:
        ValueError: On the first violation found.
    """
    n = len(targets)
    k = len(folds)
    block_size = n // k

    union: set[int] = set()
    for i, fold in enumerate(folds, start=1):
        test = set(fold["test"])
        train = set(fold["train"])

        if len(test) != block_size:
            raise ValueError(f"fold {i}: test block has {len(test)}, expected {block_size}")
        if union & test:
            raise ValueError(f"fold {i}: test block overlaps an earlier fold")
        union |= test
        if train & test:
            raise ValueError(f"fold {i}: train and test overlap")
        if train | test != set(range(n)):
            raise ValueError(f"fold {i}: train + test does not cover the dataset")

        counts: dict[int, int] = defaultdict(int)
        for idx in test:
            counts[targets[idx]] += 1
        expected = {
            label: targets.count(label) // k
            for label in set(targets)
        }
        if dict(counts) != expected:
            raise ValueError(f"fold {i}: not stratified, got {dict(counts)}, expected {expected}")

    if union != set(range(n)):
        raise ValueError("test blocks do not cover the whole dataset")
    if set(folds[0]["test"]) != set(legacy_test):
        raise ValueError("fold 1 test block does not match the legacy split")

    logger.info("All %d folds validated: disjoint, exhaustive, stratified, fold 1 == legacy", k)


def load_folds(path: str = FOLD_FILE) -> dict:
    """Load the fold manifest pickle."""
    with open(path, "rb") as f:
        return pickle.load(f)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run --with pytest,numpy pytest tests/test_folds.py -v
```

Expected: `11 passed`

- [ ] **Step 5: Commit**

```bash
git add src/data/folds.py tests/test_folds.py
git commit -m "feat: add stratified k-fold partitioning pinned to legacy ShipsNet split"
```

---

### Task 2: Generate and verify the real fold manifest

Applies Task 1's logic to the actual dataset and writes the manifest. The `fold1 == legacy` check running against real data is the gate for the entire project.

**Files:**
- Create: `scripts/make_shipsnet_folds.py`
- Reads: `datasplit/shipsnet_split_indices.pkl`, `data/shipsnet/foldered/`
- Writes: `datasplit/shipsnet_folds_k5.pkl`

**Interfaces:**
- Consumes: `build_folds`, `validate_folds`, `FOLD_FILE` from `src/data/folds.py`
- Produces: `datasplit/shipsnet_folds_k5.pkl` with structure
  `{"k": 5, "folds": [{"train": [...], "test": [...]}, ...], "meta": {...}}`

- [ ] **Step 1: Write the script**

Create `scripts/make_shipsnet_folds.py`:

```python
"""Generate the ShipsNet k=5 stratified fold manifest.

Fold 1 is pinned to the existing 80/20 split so all previously trained models
remain valid as fold-1 results.

Usage:
    uv run --with torch,torchvision,numpy python scripts/make_shipsnet_folds.py
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
```

- [ ] **Step 2: Run it**

```bash
uv run --with torch,torchvision,numpy python scripts/make_shipsnet_folds.py
```

Expected output (exact counts):

```
INFO: Dataset: 4000 images, classes=['no_ship', 'ship']
INFO: Legacy split: 3200 train / 800 test
INFO: All 5 folds validated: disjoint, exhaustive, stratified, fold 1 == legacy
INFO: Fold 1 matches the legacy split exactly (train and test)
INFO: Wrote datasplit/shipsnet_folds_k5.pkl
INFO:   fold 1: train=3200 test=800 {'no_ship': 600, 'ship': 200}
INFO:   fold 2: train=3200 test=800 {'no_ship': 600, 'ship': 200}
INFO:   fold 3: train=3200 test=800 {'no_ship': 600, 'ship': 200}
INFO:   fold 4: train=3200 test=800 {'no_ship': 600, 'ship': 200}
INFO:   fold 5: train=3200 test=800 {'no_ship': 600, 'ship': 200}
```

**STOP if any line differs.** Every downstream task assumes this exact output.

- [ ] **Step 3: Commit**

The manifest is a small deterministic artifact and must be version-controlled — results are not reproducible without it. `results/` is gitignored but `datasplit/` is not.

```bash
git add scripts/make_shipsnet_folds.py datasplit/shipsnet_folds_k5.pkl
git commit -m "feat: generate ShipsNet k=5 fold manifest with fold 1 pinned to legacy split"
```

---

### Task 3: Fold-aware data loader

**Files:**
- Modify: `src/data/shipsnet.py:22-36` (`load_data`), `:39-59` (`load_data_withval`)
- Create: `tests/test_shipsnet_loader.py`

**Interfaces:**
- Consumes: `load_folds`, `FOLD_FILE` from `src/data/folds.py`
- Produces: `load_data(batch_size: int = 16, fold: int | None = None) -> tuple[DataLoader, DataLoader]`
  and `load_data_withval(batch_size: int = 16, val_split: float = 0.1, fold: int | None = None) -> tuple[DataLoader, DataLoader, DataLoader]`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_shipsnet_loader.py`:

```python
"""Tests that load_data honours the fold argument and preserves legacy behaviour."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from src.data.shipsnet import load_data


def _test_indices(loader) -> set[int]:
    return set(loader.dataset.indices)


def test_fold_none_matches_legacy_split():
    import pickle
    _, test_loader = load_data(batch_size=16, fold=None)
    with open("datasplit/shipsnet_split_indices.pkl", "rb") as f:
        legacy = pickle.load(f)
    assert _test_indices(test_loader) == set(legacy["test"])


def test_fold_one_matches_legacy_split():
    _, legacy_test = load_data(batch_size=16, fold=None)
    _, fold1_test = load_data(batch_size=16, fold=1)
    assert _test_indices(fold1_test) == _test_indices(legacy_test)


def test_folds_have_distinct_test_sets():
    seen = []
    for fold in range(1, 6):
        _, test_loader = load_data(batch_size=16, fold=fold)
        idx = _test_indices(test_loader)
        assert len(idx) == 800
        for other in seen:
            assert idx != other, "two folds returned the same test set"
            assert not (idx & other), "fold test sets overlap"
        seen.append(idx)


def test_train_and_test_are_disjoint_per_fold():
    for fold in range(1, 6):
        train_loader, test_loader = load_data(batch_size=16, fold=fold)
        assert not (_test_indices(train_loader) & _test_indices(test_loader))
        assert len(train_loader.dataset.indices) == 3200


@pytest.mark.parametrize("bad", [0, 6, -1])
def test_invalid_fold_raises(bad):
    with pytest.raises(ValueError, match="fold"):
        load_data(batch_size=16, fold=bad)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run --with pytest,torch,torchvision,numpy pytest tests/test_shipsnet_loader.py -v
```

Expected: FAIL — `load_data() got an unexpected keyword argument 'fold'`

- [ ] **Step 3: Implement fold support**

In `src/data/shipsnet.py`, add the import below the existing ones:

```python
from src.data.folds import FOLD_FILE, load_folds
```

Then add this helper above `load_data`:

```python
def _resolve_indices(fold: int | None) -> tuple[list[int], list[int]]:
    """Return (train_indices, test_indices) for the requested fold.

    fold=None reads the legacy split file, preserving pre-k-fold behaviour exactly.

    Raises:
        ValueError: If fold is outside 1..k.
    """
    if fold is None:
        with open(SPLIT_FILE, 'rb') as f:
            split = pickle.load(f)
        return list(split['train']), list(split['test'])

    manifest = load_folds(FOLD_FILE)
    k = manifest['k']
    if not isinstance(fold, int) or not 1 <= fold <= k:
        raise ValueError(f"fold must be an integer in 1..{k}, got {fold!r}")
    entry = manifest['folds'][fold - 1]
    return list(entry['train']), list(entry['test'])
```

Replace the body of `load_data` (currently lines 22-36) with:

```python
def load_data(batch_size: int = 16, fold: int | None = None):
    """Train/test loaders. fold=None uses the legacy split; fold=1..5 uses the manifest."""
    dataset = ImageFolder(root=DATA_ROOT, transform=_base_transform())
    torch.manual_seed(42)

    train_idx, test_idx = _resolve_indices(fold)
    train_dataset = Subset(dataset, train_idx)
    test_dataset = Subset(dataset, test_idx)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=4, pin_memory=True, persistent_workers=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                             num_workers=4, pin_memory=True, persistent_workers=True)
    return train_loader, test_loader
```

Replace the split-loading portion of `load_data_withval` (currently lines 39-51) with:

```python
def load_data_withval(batch_size: int = 16, val_split: float = 0.1, fold: int | None = None):
    """Train/val/test loaders; val is carved out of the train split."""
    dataset = ImageFolder(root=DATA_ROOT, transform=_base_transform())
    torch.manual_seed(42)

    train_idx, test_idx = _resolve_indices(fold)
    full_train = Subset(dataset, train_idx)
    test_dataset = Subset(dataset, test_idx)

    train_size = int((1 - val_split) * len(full_train))
    val_size = len(full_train) - train_size
    train_dataset, val_dataset = random_split(full_train, [train_size, val_size])
```

Leave the rest of `load_data_withval` (the three `DataLoader` constructions and the `return`) unchanged.

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run --with pytest,torch,torchvision,numpy pytest tests/test_shipsnet_loader.py -v
```

Expected: `7 passed`

- [ ] **Step 5: Confirm nothing else broke**

```bash
uv run --with pyro-ppl,torch,torchvision,tqdm,scikit-learn,python-dotenv,requests,matplotlib,pandas,numpy python tests/smoke_test.py
```

Expected: `All tests passed.`

- [ ] **Step 6: Commit**

```bash
git add src/data/shipsnet.py tests/test_shipsnet_loader.py
git commit -m "feat: add fold argument to ShipsNet loaders, preserving legacy default"
```

---

### Task 4: Record fold and variant in the config JSON

The config JSON is written inside `train_svi_with_stats`, not the training script, so the metadata must be passed down. Without this, aggregation cannot tell a fold-3 smartpool run from a fold-1 base run.

**Files:**
- Modify: `src/training/svi.py:32-46` (signature), `:178-190` (config dict)
- Modify: `scripts/train_shipsnet.py:38-70` (args), `:108-145` (loop)
- Create: `tests/test_config_metadata.py`

**Interfaces:**
- Consumes: `load_data(batch_size, fold)` from Task 3
- Produces: `train_svi_with_stats(..., extra_config: dict | None = None)`. Keys in `extra_config` are merged into the saved config JSON at the top level, overriding nothing that already exists.

- [ ] **Step 1: Write the failing test**

Create `tests/test_config_metadata.py`:

```python
"""Tests that extra_config keys reach the saved config JSON."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import inspect


def test_train_svi_accepts_extra_config():
    from src.training.svi import train_svi_with_stats
    sig = inspect.signature(train_svi_with_stats)
    assert "extra_config" in sig.parameters
    assert sig.parameters["extra_config"].default is None


def test_build_config_merges_extra():
    from src.training.svi import _build_config

    base = _build_config(
        act_name="relu", prior_name="gaussian", num_epochs=100,
        best_acc=0.9, best_epoch=90, batch_size=16, train_size=3200,
        prior_mu=0.0, prior_b=1.0,
        extra_config={"fold": 3, "variant": "dropout"},
    )
    assert base["fold"] == 3
    assert base["variant"] == "dropout"
    assert base["activation"] == "relu"
    assert base["prior_params"] == {"mu": 0.0, "b": 1.0}


def test_build_config_without_extra_is_unchanged():
    from src.training.svi import _build_config

    cfg = _build_config(
        act_name="relu", prior_name="gaussian", num_epochs=100,
        best_acc=0.9, best_epoch=90, batch_size=16, train_size=3200,
        prior_mu=0.0, prior_b=1.0, extra_config=None,
    )
    assert set(cfg) == {
        "activation", "prior", "num_epochs", "best_accuracy",
        "best_accuracy_at_epoch", "batch_size", "train_size", "prior_params",
    }
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run --with pytest,pyro-ppl,torch,torchvision,tqdm,scikit-learn,matplotlib,pandas,numpy pytest tests/test_config_metadata.py -v
```

Expected: FAIL — `cannot import name '_build_config'`

- [ ] **Step 3: Extract and extend the config builder**

In `src/training/svi.py`, add this function above `train_svi_with_stats`:

```python
def _build_config(
    act_name: str,
    prior_name: str,
    num_epochs: int,
    best_acc: float,
    best_epoch: int | None,
    batch_size: int,
    train_size: int,
    prior_mu: float | None,
    prior_b: float | None,
    extra_config: dict | None = None,
) -> dict:
    """Assemble the run config JSON, merging caller-supplied metadata."""
    config = {
        'activation': act_name,
        'prior': prior_name,
        'num_epochs': num_epochs,
        'best_accuracy': best_acc,
        'best_accuracy_at_epoch': best_epoch,
        'batch_size': batch_size,
        'train_size': train_size,
        'prior_params': {'mu': prior_mu, 'b': prior_b},
    }
    if extra_config:
        config.update(extra_config)
    return config
```

Add `extra_config: dict | None = None,` as the final parameter of `train_svi_with_stats` (after `model_config_filename_pattern`).

Replace the config dict at `src/training/svi.py:178-190` with:

```python
    config = _build_config(
        act_name=act_name,
        prior_name=prior_name,
        num_epochs=num_epochs,
        best_acc=best_acc,
        best_epoch=accuracy_epochs[int(np.argmax(epoch_accuracies))] if epoch_accuracies else None,
        batch_size=train_loader.batch_size,
        train_size=len(train_loader.dataset),
        prior_mu=model.prior_mu.item() if hasattr(model, 'prior_mu') else None,
        prior_b=model.prior_b.item() if hasattr(model, 'prior_b') else None,
        extra_config=extra_config,
    )
```

- [ ] **Step 4: Run test to verify it passes**

```bash
uv run --with pytest,pyro-ppl,torch,torchvision,tqdm,scikit-learn,matplotlib,pandas,numpy pytest tests/test_config_metadata.py -v
```

Expected: `3 passed`

- [ ] **Step 5: Add `--fold` to the training script**

In `scripts/train_shipsnet.py`, add to `parse_args()` before `return parser.parse_args()`:

```python
    parser.add_argument('--fold', type=int, default=None, choices=[1, 2, 3, 4, 5],
                        help='Cross-validation fold (1-5). Omit for the legacy split.')
```

Add this helper above `main()`:

```python
def resolve_variant(args) -> str:
    """Map the mutually exclusive variant flags to the canonical variant name."""
    if args.smartpool:
        return 'smartpool'
    if args.dropout_mode:
        return 'dropout'
    if args.weight_decay:
        return 'weight_decay'
    return 'base'
```

In `main()`, replace line 136 (`train_loader, test_loader = load_data(batch_size=16)`) with:

```python
        train_loader, test_loader = load_data(batch_size=16, fold=args.fold)
```

Then extend the `train_svi_with_stats(...)` call (lines 141-145) to pass the metadata:

```python
         ts) = train_svi_with_stats(
            model, guide, svi, train_loader, device,
            num_epochs=args.epoch,
            save_dir=args.save_dir,
            extra_config={
                'fold': args.fold if args.fold is not None else 1,
                'variant': resolve_variant(args),
            },
        )
```

`fold=None` records `1` because the legacy split *is* fold 1 — this keeps the JSON schema uniform for aggregation.

- [ ] **Step 6: Verify with a trial run**

```bash
uv run --with pyro-ppl,torch,torchvision,tqdm,scikit-learn,python-dotenv,requests,matplotlib,pandas,numpy \
  python scripts/train_shipsnet.py --trial-mode --fold 2 --epoch 1 \
  --save-dir results/shipsnet/bayesian/_scratch
```

Then confirm the new keys landed:

```bash
cat results/shipsnet/bayesian/_scratch/config_*.json
```

Expected: JSON containing `"fold": 2` and `"variant": "base"` alongside the existing keys.

- [ ] **Step 7: Clean up and commit**

```bash
rm -rf results/shipsnet/bayesian/_scratch
git add src/training/svi.py scripts/train_shipsnet.py tests/test_config_metadata.py
git commit -m "feat: record fold and variant in ShipsNet training config JSON"
```

---

### Task 5: Backfill fold and variant into existing configs

The 252 existing fold-1 configs predate these keys. Aggregation needs them present everywhere.

**Files:**
- Create: `scripts/backfill_config_metadata.py`
- Create: `tests/test_backfill.py`
- Modifies in place: `results/shipsnet/bayesian/results_shipsnet_v02_0{0,1,2,3}/config_*.json`

**Interfaces:**
- Consumes: nothing from earlier tasks
- Produces: `variant_from_dirname(dirname: str) -> str | None`, `backfill_config(path: str, dry_run: bool = False) -> bool` (True if the file was or would be changed)

- [ ] **Step 1: Write the failing tests**

Create `tests/test_backfill.py`:

```python
"""Tests for config JSON backfill."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json

import pytest

from scripts.backfill_config_metadata import backfill_config, variant_from_dirname


@pytest.mark.parametrize("dirname,expected", [
    ("results_shipsnet_v02_00", "base"),
    ("results_shipsnet_v02_01", "smartpool"),
    ("results_shipsnet_v02_02", "dropout"),
    ("results_shipsnet_v02_03", "weight_decay"),
])
def test_variant_from_dirname(dirname, expected):
    assert variant_from_dirname(dirname) == expected


def test_unknown_dirname_returns_none():
    assert variant_from_dirname("results_shipsnet_old") is None


def _write(tmp_path, dirname, payload):
    d = tmp_path / dirname
    d.mkdir()
    p = d / "config_relu_gaussian_20250806_013752.json"
    p.write_text(json.dumps(payload))
    return p


def test_adds_missing_keys(tmp_path):
    p = _write(tmp_path, "results_shipsnet_v02_02", {"activation": "relu", "prior": "gaussian"})
    assert backfill_config(str(p)) is True
    result = json.loads(p.read_text())
    assert result["fold"] == 1
    assert result["variant"] == "dropout"
    assert result["activation"] == "relu"


def test_is_idempotent(tmp_path):
    p = _write(tmp_path, "results_shipsnet_v02_00", {"activation": "relu"})
    assert backfill_config(str(p)) is True
    assert backfill_config(str(p)) is False


def test_dry_run_does_not_write(tmp_path):
    p = _write(tmp_path, "results_shipsnet_v02_00", {"activation": "relu"})
    assert backfill_config(str(p), dry_run=True) is True
    assert "fold" not in json.loads(p.read_text())


def test_unknown_variant_raises(tmp_path):
    p = _write(tmp_path, "results_shipsnet_mystery", {"activation": "relu"})
    with pytest.raises(ValueError, match="variant"):
        backfill_config(str(p))
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run --with pytest pytest tests/test_backfill.py -v
```

Expected: collection error — `No module named 'scripts.backfill_config_metadata'`

- [ ] **Step 3: Implement the backfill script**

First make `scripts/` importable, so the test can `from scripts.backfill_config_metadata import ...`. The file must exist but stay empty — do **not** add re-exports, as the existing scripts are run directly and importing them at package level would execute their heavy dependencies.

```bash
touch scripts/__init__.py
```

Create `scripts/backfill_config_metadata.py`:

```python
"""Stamp fold and variant into pre-k-fold ShipsNet config JSONs.

Existing runs were trained on the legacy 80/20 split, which is fold 1 of the
k=5 manifest, so they are tagged fold=1. The variant is recovered from the
containing directory name.

Usage:
    uv run python scripts/backfill_config_metadata.py --dry-run
    uv run python scripts/backfill_config_metadata.py
"""
import argparse
import json
import logging
import os
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DIR_TO_VARIANT = {
    "results_shipsnet_v02_00": "base",
    "results_shipsnet_v02_01": "smartpool",
    "results_shipsnet_v02_02": "dropout",
    "results_shipsnet_v02_03": "weight_decay",
}

DEFAULT_ROOT = "results/shipsnet/bayesian"


def variant_from_dirname(dirname: str) -> str | None:
    """Map a results directory name to its canonical variant, or None if unknown."""
    return DIR_TO_VARIANT.get(dirname)


def backfill_config(path: str, dry_run: bool = False) -> bool:
    """Add fold and variant to one config JSON.

    Returns:
        True if the file was changed (or would be, when dry_run).

    Raises:
        ValueError: If the containing directory maps to no known variant.
    """
    p = Path(path)
    variant = variant_from_dirname(p.parent.name)
    if variant is None:
        raise ValueError(f"cannot determine variant for directory {p.parent.name!r}")

    with open(p) as f:
        config = json.load(f)

    if config.get("fold") == 1 and config.get("variant") == variant:
        return False

    config["fold"] = 1
    config["variant"] = variant
    if not dry_run:
        with open(p, "w") as f:
            json.dump(config, f, indent=4)
    return True


def parse_args():
    parser = argparse.ArgumentParser(description="Backfill fold/variant into config JSONs")
    parser.add_argument("--root", type=str, default=DEFAULT_ROOT)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    changed = skipped = 0
    for dirname in DIR_TO_VARIANT:
        d = Path(args.root) / dirname
        if not d.is_dir():
            logger.warning("missing directory: %s", d)
            continue
        configs = sorted(d.glob("config_*.json"))
        logger.info("%s: %d configs (variant=%s)", dirname, len(configs), DIR_TO_VARIANT[dirname])
        for cfg in configs:
            if backfill_config(str(cfg), dry_run=args.dry_run):
                changed += 1
            else:
                skipped += 1
    verb = "would change" if args.dry_run else "changed"
    logger.info("Done: %s %d, already correct %d", verb, changed, skipped)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run --with pytest pytest tests/test_backfill.py -v
```

Expected: `9 passed`

- [ ] **Step 5: Dry-run against the real configs**

```bash
uv run python scripts/backfill_config_metadata.py --dry-run
```

Expected: four directory lines summing to roughly 253 configs, then
`INFO: Done: would change 253, already correct 0`

- [ ] **Step 6: Apply, then spot-check**

```bash
uv run python scripts/backfill_config_metadata.py
cat results/shipsnet/bayesian/results_shipsnet_v02_02/config_*.json | head -20
```

Expected: `"fold": 1` and `"variant": "dropout"` present.

Confirm idempotency:

```bash
uv run python scripts/backfill_config_metadata.py
```

Expected: `INFO: Done: changed 0, already correct 253`

- [ ] **Step 7: Commit**

`results/` is gitignored, so only the scripts are committed.

```bash
git add scripts/__init__.py scripts/backfill_config_metadata.py tests/test_backfill.py
git commit -m "feat: backfill fold and variant into pre-k-fold ShipsNet configs"
```

---

### Task 6: Fold support in SEU evaluation

The highest-risk change in the project. A fold-3 model evaluated against fold 1's test set is being scored on images it trained on — producing optimistic numbers with no error and no warning.

**Files:**
- Modify: `scripts/eval_seu_shipsnet.py:44-56` (args), `:228` (data load), `:298-316` (result rows)
- Create: `tests/test_seu_fold_wiring.py`

**Interfaces:**
- Consumes: `load_data(batch_size, fold)` from Task 3; `fold`/`variant` config keys from Tasks 4-5
- Produces: SEU CSVs gaining two columns, `fold` and `variant`

- [ ] **Step 1: Write the failing test**

Create `tests/test_seu_fold_wiring.py`:

```python
"""Guards the fold wiring in the SEU script.

These are source-level assertions rather than end-to-end runs: a full SEU sweep
takes ~5 minutes on a GPU, which is too slow for a unit test, but a silently
wrong test set is the worst possible failure here, so the wiring is checked
directly.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import ast

SCRIPT = Path(__file__).parent.parent / "scripts" / "eval_seu_shipsnet.py"


def _source() -> str:
    return SCRIPT.read_text(encoding="utf-8")


def test_declares_fold_argument():
    assert "'--fold'" in _source() or '"--fold"' in _source()


def test_load_data_receives_fold():
    """load_data must be called with fold=..., never bare."""
    tree = ast.parse(_source())
    calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "load_data"
    ]
    assert calls, "load_data is never called"
    for call in calls:
        kwargs = {kw.arg for kw in call.keywords}
        assert "fold" in kwargs, "load_data called without fold= — SEU would use the wrong test set"


def test_result_rows_carry_fold_and_variant():
    src = _source()
    assert '"fold":' in src or "'fold':" in src
    assert '"variant":' in src or "'variant':" in src
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run --with pytest pytest tests/test_seu_fold_wiring.py -v
```

Expected: FAIL on all three — no `--fold` argument exists yet.

- [ ] **Step 3: Wire fold through the SEU script**

In `parse_args()` of `scripts/eval_seu_shipsnet.py`, add before `return parser.parse_args()`:

```python
    parser.add_argument('--fold', type=int, default=None, choices=[1, 2, 3, 4, 5],
                        help='Cross-validation fold (1-5) the models were trained on. '
                             'Must match the fold in the model configs.')
```

Replace line 228 (`_, test_loader = load_data(batch_size=16)`) with:

```python
    _, test_loader = load_data(batch_size=16, fold=args.fold)
```

Add this guard immediately after, so a mismatched `--fold` fails loudly instead of producing quiet nonsense:

```python
    def _assert_fold_matches(config: dict, ts: str) -> None:
        """Refuse to evaluate a model against a test set it may have trained on."""
        expected = args.fold if args.fold is not None else 1
        actual = config.get('fold')
        if actual is None:
            raise ValueError(
                f"config for {ts} has no 'fold' key — run "
                f"scripts/backfill_config_metadata.py first"
            )
        if actual != expected:
            raise ValueError(
                f"fold mismatch for {ts}: model trained on fold {actual}, "
                f"but --fold={expected} selected the fold-{expected} test set. "
                f"This would evaluate on training images."
            )
```

In the per-model loop, immediately after `prior = model_config['prior']` (line 279), add:

```python
        _assert_fold_matches(model_config, ts)
```

Finally, add the two columns to the result row dict (inside the `results.append({...})` block at lines 298-316), directly after `"prior": prior,`:

```python
                                "fold": model_config.get('fold', 1),
                                "variant": model_config.get('variant', 'base'),
```

- [ ] **Step 4: Run test to verify it passes**

```bash
uv run --with pytest pytest tests/test_seu_fold_wiring.py -v
```

Expected: `3 passed`

- [ ] **Step 5: Commit**

```bash
git add scripts/eval_seu_shipsnet.py tests/test_seu_fold_wiring.py
git commit -m "feat: add fold support and mismatch guard to ShipsNet SEU evaluation"
```

---

### Task 7: Cross-fold aggregation

No aggregation code exists in the repository — the paper's current tables were built outside version control. Without this, the sweep produces ~1,400 CSVs and no tables.

**Files:**
- Create: `src/evaluation/aggregate.py`
- Create: `scripts/aggregate_seu.py`
- Create: `tests/test_aggregate.py`

**Interfaces:**
- Consumes: `aggregate_robustness_index` from `src/evaluation/metrics.py`; SEU CSVs carrying `fold` and `variant` from Task 6
- Produces:
  - `load_seu_results(root: str) -> pd.DataFrame`
  - `add_robustness_metrics(df: pd.DataFrame) -> pd.DataFrame` — adds `aad`, `arin`
  - `aggregate_across_folds(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame` — adds `<metric>_mean`, `<metric>_std`, `n_folds`, `n_excluded`
  - `to_latex(df: pd.DataFrame, caption: str, label: str) -> str`
  - `GROUP_PRESETS: dict[str, list[str]]`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_aggregate.py`:

```python
"""Tests for cross-fold SEU aggregation."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import pytest

from src.evaluation.aggregate import (
    add_robustness_metrics,
    aggregate_across_folds,
    to_latex,
)


def make_df() -> pd.DataFrame:
    """Two folds of one config; accuracy_change differs so std is non-zero."""
    rows = []
    for fold, acc_change, smd in [(1, -0.10, 0.20), (2, -0.20, 0.40)]:
        rows.append({
            "fold": fold, "variant": "base", "activation_fn": "relu",
            "prior": "gaussian", "prior_b": 1.0, "param_type": "locs",
            "location_layer": "conv1", "location_module": "weight", "bit_index": 0,
            "initial_accuracy": 0.90, "accuracy_change": acc_change,
            "softmax_difference": smd, "remarks": "",
        })
    return pd.DataFrame(rows)


def test_adds_aad_as_absolute_value():
    df = add_robustness_metrics(make_df())
    assert list(df["aad"]) == [0.10, 0.20]


def test_adds_arin_as_rms():
    df = add_robustness_metrics(make_df())
    expected = np.sqrt((0.10 ** 2 + 0.20 ** 2) / 2)
    assert abs(df["arin"].iloc[0] - expected) < 1e-9


def test_aggregates_mean_and_std_across_folds():
    df = add_robustness_metrics(make_df())
    out = aggregate_across_folds(df, ["variant", "activation_fn", "prior"])
    assert len(out) == 1
    row = out.iloc[0]
    assert abs(row["aad_mean"] - 0.15) < 1e-9
    assert abs(row["aad_std"] - np.std([0.10, 0.20], ddof=1)) < 1e-9
    assert row["n_folds"] == 2


def test_excludes_nan_rows_and_counts_them():
    df = add_robustness_metrics(make_df())
    df.loc[1, "softmax_difference"] = np.nan
    df.loc[1, "accuracy_change"] = np.nan
    df = add_robustness_metrics(df)
    out = aggregate_across_folds(df, ["variant"])
    row = out.iloc[0]
    assert row["n_folds"] == 1
    assert row["n_excluded"] == 1


def test_all_nan_group_yields_zero_folds_not_crash():
    df = add_robustness_metrics(make_df())
    df["accuracy_change"] = np.nan
    df["softmax_difference"] = np.nan
    df = add_robustness_metrics(df)
    out = aggregate_across_folds(df, ["variant"])
    assert out.iloc[0]["n_folds"] == 0
    assert out.iloc[0]["n_excluded"] == 2


def test_latex_output_contains_table_scaffolding():
    df = add_robustness_metrics(make_df())
    out = aggregate_across_folds(df, ["variant"])
    tex = to_latex(out, caption="Test", label="tab:test")
    assert "\\begin{table}" in tex
    assert "\\caption{Test}" in tex
    assert "\\label{tab:test}" in tex
    assert "resizebox" in tex, "must constrain width — reviewer flagged out-of-bounds tables"


def test_missing_group_column_raises():
    df = add_robustness_metrics(make_df())
    with pytest.raises(KeyError):
        aggregate_across_folds(df, ["nonexistent_column"])
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run --with pytest,pandas,numpy,torch pytest tests/test_aggregate.py -v
```

Expected: collection error — `No module named 'src.evaluation.aggregate'`

- [ ] **Step 3: Implement the aggregation module**

Create `src/evaluation/aggregate.py`:

```python
"""Reduce per-fold SEU result CSVs into mean +/- std tables for the paper.

Metric definitions follow src/evaluation/metrics.py:
    AAD  = |accuracy_after - accuracy_before|
    ARIn = sqrt((AAD^2 + SoftmaxDiff^2) / 2)
"""
import glob
import logging
import os

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

METRICS = ["aad", "softmax_difference", "arin", "initial_accuracy"]

GROUP_PRESETS: dict[str, list[str]] = {
    # Table 1 style: one row per model variant, averaged over everything else.
    "variant": ["variant"],
    # Per design-choice breakdown.
    "design": ["variant", "activation_fn", "prior", "prior_b"],
    # Full per-injection-site breakdown (Tables 3-5 style).
    "site": ["variant", "activation_fn", "prior", "prior_b",
             "param_type", "location_layer", "location_module", "bit_index"],
}


def load_seu_results(root: str) -> pd.DataFrame:
    """Recursively load every SEU CSV under root into one DataFrame.

    Raises:
        FileNotFoundError: If no CSVs are found.
    """
    paths = sorted(glob.glob(os.path.join(root, "**", "*.csv"), recursive=True))
    if not paths:
        raise FileNotFoundError(f"no SEU CSVs found under {root}")

    frames = []
    for p in paths:
        df = pd.read_csv(p)
        df["source_file"] = os.path.relpath(p, root)
        frames.append(df)
    combined = pd.concat(frames, ignore_index=True)
    logger.info("Loaded %d rows from %d files", len(combined), len(paths))
    return combined


def add_robustness_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Add aad and arin columns. NaN inputs propagate to NaN outputs."""
    out = df.copy()
    out["aad"] = out["accuracy_change"].abs()
    out["arin"] = np.sqrt((out["aad"] ** 2 + out["softmax_difference"] ** 2) / 2)
    return out


def aggregate_across_folds(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    """Reduce to mean/std per group, excluding degenerate (NaN) injections.

    Degenerate injections are those the SEU script marked NaN (negative
    scale/width). They are excluded from statistics and counted in n_excluded
    rather than silently dropped.

    Raises:
        KeyError: If any group column is absent.
    """
    missing = [c for c in group_cols if c not in df.columns]
    if missing:
        raise KeyError(f"group columns not in DataFrame: {missing}")

    valid = df["arin"].notna()
    records = []
    for keys, group in df.groupby(group_cols, dropna=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        good = group[valid.loc[group.index]]
        record = dict(zip(group_cols, keys))
        for metric in METRICS:
            if metric not in group.columns:
                continue
            series = good[metric].dropna()
            record[f"{metric}_mean"] = series.mean() if len(series) else np.nan
            record[f"{metric}_std"] = series.std(ddof=1) if len(series) > 1 else 0.0
        record["n_folds"] = int(good["fold"].nunique()) if "fold" in good.columns else len(good)
        record["n_excluded"] = int(len(group) - len(good))
        records.append(record)

    return pd.DataFrame(records)


def to_latex(df: pd.DataFrame, caption: str, label: str, float_fmt: str = "%.4f") -> str:
    """Render a LaTeX table wrapped in resizebox to fit ICAART column width.

    Reviewer #1 flagged Tables 3-5 as running out of page bounds, so width is
    constrained rather than left to the default.
    """
    body = df.to_latex(index=False, float_format=float_fmt, escape=True, longtable=False)
    return (
        "\\begin{table}[t]\n"
        "\\centering\n"
        f"\\caption{{{caption}}}\n"
        f"\\label{{{label}}}\n"
        "\\resizebox{\\columnwidth}{!}{%\n"
        f"{body}"
        "}\n"
        "\\end{table}\n"
    )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run --with pytest,pandas,numpy,torch pytest tests/test_aggregate.py -v
```

Expected: `7 passed`

- [ ] **Step 5: Add the CLI wrapper**

Create `scripts/aggregate_seu.py`:

```python
"""Aggregate ShipsNet SEU results across folds into paper-ready tables.

Usage:
    uv run --with pandas,numpy python scripts/aggregate_seu.py \\
        --root results/shipsnet/seu --preset variant --out results/tables
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import logging

from src.evaluation.aggregate import (
    GROUP_PRESETS,
    add_robustness_metrics,
    aggregate_across_folds,
    load_seu_results,
    to_latex,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Aggregate SEU results across folds")
    parser.add_argument("--root", type=str, default="results/shipsnet/seu")
    parser.add_argument("--preset", type=str, default="variant", choices=sorted(GROUP_PRESETS))
    parser.add_argument("--out", type=str, default="results/tables")
    parser.add_argument("--caption", type=str, default="SEU robustness, mean $\\pm$ std across 5 folds")
    parser.add_argument("--label", type=str, default="tab:seu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = load_seu_results(args.root)
    df = add_robustness_metrics(df)

    group_cols = GROUP_PRESETS[args.preset]
    agg = aggregate_across_folds(df, group_cols)

    incomplete = agg[agg["n_folds"] < 5]
    if len(incomplete):
        logger.warning("%d/%d groups have fewer than 5 folds", len(incomplete), len(agg))

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"seu_{args.preset}.csv"
    tex_path = out_dir / f"seu_{args.preset}.tex"

    agg.to_csv(csv_path, index=False)
    tex_path.write_text(to_latex(agg, caption=args.caption, label=args.label), encoding="utf-8")

    logger.info("Wrote %s and %s (%d groups)", csv_path, tex_path, len(agg))
    logger.info("Total degenerate injections excluded: %d", int(agg["n_excluded"].sum()))


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Smoke-run against existing fold-1 data**

Existing SEU CSVs lack `fold`/`variant` columns, so this must be run against a fold-aware directory. Confirm the failure mode is clear:

```bash
uv run --with pandas,numpy python scripts/aggregate_seu.py \
  --root results/shipsnet/seu/results_shipsnet_v02_00_SEU --preset design
```

Expected: `KeyError: group columns not in DataFrame: ['variant']` — correct, since these predate Task 6. This confirms the guard works; real aggregation happens in Task 8.

- [ ] **Step 7: Commit**

```bash
git add src/evaluation/aggregate.py scripts/aggregate_seu.py tests/test_aggregate.py
git commit -m "feat: add cross-fold SEU aggregation with mean/std and LaTeX output"
```

---

### Task 8: End-to-end pilot

The gate before ~10 days of GPU time. One config, all four new folds, train → SEU → aggregate. Its purpose is to catch naming, wiring, and grouping bugs while they cost an hour instead of nine days.

**Files:**
- Creates: `results/shipsnet/bayesian/fold{2,3,4,5}/_pilot/`, `results/shipsnet/seu/fold{2,3,4,5}/_pilot/`
- No source changes expected — if any are needed, that is the pilot doing its job.

**Interfaces:**
- Consumes: everything from Tasks 1-7

- [ ] **Step 1: Train one config on folds 2-5**

```bash
for FOLD in 2 3 4 5; do
  uv run --with pyro-ppl,torch,torchvision,tqdm,scikit-learn,python-dotenv,requests,matplotlib,pandas,numpy \
    python scripts/train_shipsnet.py \
      --prior Gaussian_prior --b-set single --epoch 5 --fold $FOLD \
      --save-dir results/shipsnet/bayesian/fold$FOLD/_pilot
done
```

`--b-set single` restricts to b=1.0; `--epoch 5` keeps the pilot short. Expect 7 activations × 1 b = 7 runs per fold.

- [ ] **Step 2: Verify fold and variant landed in every config**

```bash
uv run python -c "
import json, glob
for f in sorted(glob.glob('results/shipsnet/bayesian/fold*/_pilot/config_*.json')):
    c = json.load(open(f))
    print(f.split('/')[-2], c['fold'], c['variant'], c['activation'], c['train_size'])
"
```

Expected: 28 lines, `train_size` 3200 on every line, `fold` matching the directory, `variant` always `base`.

- [ ] **Step 3: Confirm the folds actually differ**

The single most important check in the plan — that folds are not silently identical.

```bash
uv run --with torch,torchvision,numpy python -c "
from src.data.shipsnet import load_data
seen = {}
for fold in range(1, 6):
    _, test = load_data(batch_size=16, fold=fold)
    idx = frozenset(test.dataset.indices)
    seen[fold] = idx
    print(f'fold {fold}: {len(idx)} test indices, first 5 = {sorted(idx)[:5]}')
assert len(set(seen.values())) == 5, 'FOLDS ARE NOT DISTINCT'
union = set().union(*seen.values())
assert len(union) == 4000, f'union is {len(union)}, expected 4000'
print('OK: 5 distinct folds covering all 4000 indices')
"
```

Expected: five distinct index sets, ending `OK: 5 distinct folds covering all 4000 indices`

- [ ] **Step 4: Run SEU on the pilot models**

```bash
for FOLD in 2 3 4 5; do
  uv run --with pyro-ppl,torch,torchvision,tqdm,scikit-learn,python-dotenv,requests,matplotlib,pandas,numpy \
    python scripts/eval_seu_shipsnet.py \
      --prior Gaussian_prior --fold $FOLD \
      --search-dir results/shipsnet/bayesian/fold$FOLD/_pilot \
      --save-dir results/shipsnet/seu/fold$FOLD/_pilot
done
```

- [ ] **Step 5: Verify the fold mismatch guard actually fires**

Deliberately point a fold-2 model at the fold-3 test set. This *must* fail.

```bash
uv run --with pyro-ppl,torch,torchvision,tqdm,scikit-learn,python-dotenv,requests,matplotlib,pandas,numpy \
  python scripts/eval_seu_shipsnet.py \
    --prior Gaussian_prior --fold 3 \
    --search-dir results/shipsnet/bayesian/fold2/_pilot \
    --save-dir results/shipsnet/seu/_guardcheck
```

Expected: `ValueError: fold mismatch for <ts>: model trained on fold 2, but --fold=3 selected the fold-3 test set. This would evaluate on training images.`

**If this command succeeds instead of failing, stop.** The guard is broken and the entire sweep would be at risk of silent contamination.

```bash
rm -rf results/shipsnet/seu/_guardcheck
```

- [ ] **Step 6: Aggregate the pilot**

```bash
uv run --with pandas,numpy python scripts/aggregate_seu.py \
  --root results/shipsnet/seu --preset design --out results/tables/_pilot
```

Expected: no `KeyError`; log reports the group count and excluded-injection total. Groups will report `n_folds` of 4 (folds 2-5 only) — fold 1 has no fold-aware CSVs yet.

- [ ] **Step 7: Inspect the aggregated output**

```bash
uv run --with pandas python -c "
import pandas as pd
df = pd.read_csv('results/tables/_pilot/seu_design.csv')
print(df[['variant','activation_fn','prior','prior_b','aad_mean','aad_std','arin_mean','arin_std','n_folds','n_excluded']].to_string())
"
```

Expected: 7 rows (one per activation), `n_folds` = 4, and **non-zero `aad_std`** — a std of exactly 0.0 everywhere would mean the folds produced identical results, i.e. the fold argument is not reaching training.

- [ ] **Step 8: Clean up and commit any fixes**

```bash
rm -rf results/shipsnet/bayesian/fold*/_pilot results/shipsnet/seu/fold*/_pilot results/tables/_pilot
```

If the pilot required source changes:

```bash
git add -A
git commit -m "fix: address issues found in k-fold pilot run"
```

---

### Task 9: Production sweep

Long-running and resumable. Not TDD — this is execution.

**Files:**
- Writes: `results/shipsnet/bayesian/fold{2,3,4,5}/results_shipsnet_v02_0{0,1,2,3}/`
- Writes: `results/shipsnet/seu/fold{1,2,3,4,5}/results_shipsnet_v02_0{0,1,2,3}/`

- [ ] **Step 1: Launch training, folds 2-5, full grid**

252 configs × 4 folds = 1008 runs at ~8 min ≈ **5.6 days**. Run per (fold, variant, prior) so failures are isolated and resumable.

```bash
for FOLD in 2 3 4 5; do
  for VARIANT in "00:" "01:--smartpool" "02:--dropout-mode" "03:--wd"; do
    IDX="${VARIANT%%:*}"; FLAG="${VARIANT#*:}"
    for PRIOR in Gaussian_prior Laplace_prior Uniform_prior; do
      uv run --with pyro-ppl,torch,torchvision,tqdm,scikit-learn,python-dotenv,requests,matplotlib,pandas,numpy \
        python scripts/train_shipsnet.py \
          --prior $PRIOR --epoch 100 --b-set full --fold $FOLD $FLAG \
          --save-dir results/shipsnet/bayesian/fold$FOLD/results_shipsnet_v02_$IDX
    done
  done
done
```

- [ ] **Step 2: Verify training completeness before starting SEU**

```bash
uv run python -c "
import glob, json, collections
counts = collections.Counter()
for f in glob.glob('results/shipsnet/bayesian/fold*/results_shipsnet_v02_*/config_*.json'):
    c = json.load(open(f))
    counts[(c['fold'], c['variant'])] += 1
for key in sorted(counts):
    print(key, counts[key])
print('total:', sum(counts.values()))
"
```

Expected: 16 rows (4 folds × 4 variants), 63 each, total 1008. Investigate any shortfall before proceeding — SEU on a partial grid wastes days.

- [ ] **Step 3: Run SEU on folds 2-5**

1008 sweeps at ~5 min ≈ **3.5 days**.

```bash
for FOLD in 2 3 4 5; do
  for IDX in 00 01 02 03; do
    for PRIOR in Gaussian_prior Laplace_prior Uniform_prior; do
      uv run --with pyro-ppl,torch,torchvision,tqdm,scikit-learn,python-dotenv,requests,matplotlib,pandas,numpy \
        python scripts/eval_seu_shipsnet.py \
          --prior $PRIOR --fold $FOLD \
          --search-dir results/shipsnet/bayesian/fold$FOLD/results_shipsnet_v02_$IDX \
          --save-dir results/shipsnet/seu/fold$FOLD/results_shipsnet_v02_$IDX
    done
  done
done
```

- [ ] **Step 4: Complete fold-1 SEU coverage**

Existing fold-1 SEU covers b=1.0 only. The full grid needs the b=10.0 and b=0.1 configs — ~168 additional sweeps ≈ **14 hours**. The script skips timestamps whose CSV already exists, so completed b=1.0 sweeps are not repeated.

```bash
for IDX in 00 01 02 03; do
  for PRIOR in Gaussian_prior Laplace_prior Uniform_prior; do
    uv run --with pyro-ppl,torch,torchvision,tqdm,scikit-learn,python-dotenv,requests,matplotlib,pandas,numpy \
      python scripts/eval_seu_shipsnet.py \
        --prior $PRIOR --fold 1 \
        --search-dir results/shipsnet/bayesian/results_shipsnet_v02_$IDX \
        --save-dir results/shipsnet/seu/fold1/results_shipsnet_v02_$IDX
  done
done
```

- [ ] **Step 5: Generate the final tables**

```bash
for PRESET in variant design site; do
  uv run --with pandas,numpy python scripts/aggregate_seu.py \
    --root results/shipsnet/seu --preset $PRESET --out results/tables \
    --caption "ShipsNet SEU robustness by $PRESET, mean \$\\pm\$ std over 5 folds" \
    --label "tab:seu-$PRESET"
done
```

- [ ] **Step 6: Verify every group has all 5 folds**

```bash
uv run --with pandas python -c "
import pandas as pd
df = pd.read_csv('results/tables/seu_design.csv')
bad = df[df['n_folds'] != 5]
print('groups:', len(df), 'incomplete:', len(bad))
if len(bad): print(bad.to_string())
print('total excluded injections:', int(df['n_excluded'].sum()))
"
```

Expected: `incomplete: 0`. Any incomplete group means missing runs, and its std is not trustworthy.

- [ ] **Step 7: Commit the tables**

`results/` is gitignored, so force-add the small table artifacts.

```bash
git add -f results/tables/seu_variant.csv results/tables/seu_design.csv results/tables/seu_site.csv
git add -f results/tables/seu_variant.tex results/tables/seu_design.tex results/tables/seu_site.tex
git commit -m "results: add ShipsNet 5-fold SEU aggregation tables"
```

---

## Notes for the paper

- Accuracy figures must come from the SEU CSVs' `initial_accuracy` column (best checkpoint, MC-10, correct fold test set), **not** the training-side `test_acc`, which is last-epoch. Mixing them would inflate the reported std with a bookkeeping artifact. See spec §6.2.
- `n_excluded` should be reported alongside the tables — it is the count of degenerate injections (negative scale/width) excluded from statistics, and it also settles Reviewer #1's query about the inconsistent SEU injection counts (15,456 vs 13,104).
- This plan covers ShipsNet only. EuroSAT k-fold and the EuroSAT SEU sweep remain separate, unstarted work.
