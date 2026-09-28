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
