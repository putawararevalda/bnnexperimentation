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
