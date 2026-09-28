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
