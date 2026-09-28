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
