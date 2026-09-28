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
