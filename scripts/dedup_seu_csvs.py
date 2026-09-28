"""Deduplicate SEU result CSVs, writing cleaned copies to a separate tree.

Why this exists
---------------
Some SEU result CSVs contain more than one evaluation pass appended into a
single file: an earlier version of the eval scripts opened the CSV in append
mode with no per-combo resume, so re-running a config added a second set of
rows rather than skipping the finished work.

The duplicates are invisible to the completeness check in the eval scripts,
because ``load_completed_combos()`` returns a *set* of combo keys -- a file
with 336 rows covering the same 168 combos looks "done". They are NOT
invisible to ``src/evaluation/aggregate.py:load_seu_results()``, which
concatenates every row, so affected configs are silently double-weighted in
any mean/std over the grid.

What a "pass" is
----------------
Every row carries the ``initial_accuracy`` of the model it was measured
against. One injector construction = one baseline MC evaluation = one value.
Rows therefore partition into passes by ``initial_accuracy``, and mixing rows
from two passes in one config would pair flip results against the wrong
baseline. This script keeps exactly one pass per file.

Selection policy: prefer a pass that covers all expected combos; break ties
with --keep (first = earliest in file order, the default).

Output
------
Cleaned copies are written under --out-dir, mirroring the --root layout, so
the result is a drop-in replacement tree for the aggregator. Files that need
no change are copied byte-for-byte. Originals are never modified.

Kept rows are copied as raw text rather than re-serialised through pandas, so
float formatting is preserved exactly (a CSV round-trip drops the last digit
of a float repr).

Usage:
    # audit only -- report what would change, write nothing
    uv run --with pandas python scripts/dedup_seu_csvs.py \\
        --root results/eurosat/seu

    # write the cleaned tree
    uv run --with pandas python scripts/dedup_seu_csvs.py \\
        --root results/eurosat/seu --out-dir results/eurosat/seu_clean --apply
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import csv
import io
import logging
import os
import shutil
from typing import Dict, List, Optional, Sequence, Tuple

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Identifies one SEU injection within a config.
COMBO_KEY: Tuple[str, ...] = (
    "param_type", "location_index", "location_layer", "location_module", "bit_index",
)
PASS_KEY = "initial_accuracy"
DEFAULT_EXPECTED_COMBOS = 168


class DedupError(RuntimeError):
    """Raised when a CSV cannot be deduplicated safely."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Deduplicate SEU result CSVs into a cleaned copy tree")
    parser.add_argument("--root", type=str, required=True,
                        help="Directory to scan recursively for SEU CSVs.")
    parser.add_argument("--out-dir", type=str, default=None,
                        help="Where to write the cleaned tree. Required with --apply.")
    parser.add_argument("--apply", action="store_true",
                        help="Actually write files. Without it, audit only.")
    parser.add_argument("--keep", type=str, default="first", choices=["first", "last"],
                        help="Which qualifying pass to keep (default: first).")
    parser.add_argument("--expected-combos", type=int, default=DEFAULT_EXPECTED_COMBOS,
                        help=f"Combos per complete config (default: {DEFAULT_EXPECTED_COMBOS}).")
    return parser.parse_args()


def read_rows(path: str) -> Tuple[str, List[str], List[Dict[str, str]]]:
    """Return (header_line, data_lines, parsed_rows) for a CSV.

    `data_lines` are the raw source lines, positionally aligned with
    `parsed_rows`, so selected rows can be re-emitted verbatim.

    Raises:
        DedupError: If the file is empty, or a field contains an embedded
            newline (which would break the line/row alignment).
    """
    with open(path, "r", newline="") as f:
        lines = f.readlines()
    if not lines:
        raise DedupError(f"{path}: empty file")

    header_line, data_lines = lines[0], lines[1:]
    rows = list(csv.DictReader(io.StringIO("".join(lines))))
    if len(rows) != len(data_lines):
        raise DedupError(
            f"{path}: {len(rows)} parsed rows vs {len(data_lines)} source lines "
            "(embedded newline in a field?); refusing to guess alignment")
    return header_line, data_lines, rows


def combo_of(row: Dict[str, str]) -> Tuple[str, ...]:
    """Extract the combo key from a parsed row.

    Raises:
        DedupError: If a key column is missing.
    """
    try:
        return tuple(row[c] for c in COMBO_KEY)
    except KeyError as e:
        raise DedupError(f"row missing column {e}; not an SEU result CSV?") from e


def split_passes(rows: Sequence[Dict[str, str]]) -> List[Tuple[str, List[int]]]:
    """Group row indices into passes by initial_accuracy, in order of first
    appearance. Returns a list of (initial_accuracy, row_indices)."""
    order: List[str] = []
    groups: Dict[str, List[int]] = {}
    for i, row in enumerate(rows):
        value = row.get(PASS_KEY, "")
        if value not in groups:
            groups[value] = []
            order.append(value)
        groups[value].append(i)
    return [(v, groups[v]) for v in order]


def choose_pass(passes: Sequence[Tuple[str, List[int]]],
                rows: Sequence[Dict[str, str]],
                expected: int,
                keep: str) -> Tuple[str, List[int]]:
    """Pick the pass to retain: a complete one if available, else the largest.

    Raises:
        DedupError: If there are no passes.
    """
    if not passes:
        raise DedupError("no rows to choose from")

    def n_combos(indices: Sequence[int]) -> int:
        return len({combo_of(rows[i]) for i in indices})

    complete = [p for p in passes if n_combos(p[1]) >= expected]
    candidates = complete or list(passes)
    if not complete:
        # Nothing covers the full grid; keep the widest so we discard least.
        candidates = sorted(candidates, key=lambda p: n_combos(p[1]), reverse=True)
        return candidates[0]
    return candidates[-1] if keep == "last" else candidates[0]


def dedup_indices(indices: Sequence[int],
                  rows: Sequence[Dict[str, str]]) -> List[int]:
    """Drop repeated combos within a single pass, keeping first occurrence.

    Only fires when two passes happened to share an initial_accuracy and so
    were grouped together; normally this is a no-op.
    """
    seen = set()
    kept = []
    for i in indices:
        combo = combo_of(rows[i])
        if combo in seen:
            continue
        seen.add(combo)
        kept.append(i)
    return kept


def plan_file(path: str, expected: int, keep: str) -> Optional[Dict]:
    """Analyse one CSV. Returns a plan dict, or None if it needs no change.

    Raises:
        DedupError: If the file cannot be parsed safely.
    """
    header_line, data_lines, rows = read_rows(path)
    combos = [combo_of(r) for r in rows]
    if len(combos) == len(set(combos)):
        return None

    passes = split_passes(rows)
    chosen_value, chosen_indices = choose_pass(passes, rows, expected, keep)
    kept = dedup_indices(chosen_indices, rows)

    return {
        "path": path,
        "header_line": header_line,
        "data_lines": data_lines,
        "rows": rows,
        "n_original": len(rows),
        "n_kept": len(kept),
        "kept_indices": kept,
        "chosen_pass": chosen_value,
        "passes": [(v, len(ix), len({combo_of(rows[i]) for i in ix})) for v, ix in passes],
        "complete": len({combo_of(rows[i]) for i in kept}) >= expected,
    }


def write_cleaned(plan: Dict, dest: str) -> None:
    """Write the retained rows to `dest`, then verify the result.

    Raises:
        DedupError: If the written file fails verification.
    """
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    body = [plan["data_lines"][i] for i in plan["kept_indices"]]
    if body and not body[-1].endswith("\n"):
        body[-1] += "\n"
    with open(dest, "w", newline="") as f:
        f.write(plan["header_line"])
        f.writelines(body)

    _, out_lines, out_rows = read_rows(dest)
    out_combos = [combo_of(r) for r in out_rows]
    if len(out_combos) != len(set(out_combos)):
        raise DedupError(f"{dest}: still has duplicate combos after cleaning")
    if len(out_rows) != plan["n_kept"]:
        raise DedupError(f"{dest}: wrote {len(out_rows)} rows, expected {plan['n_kept']}")
    if len({r.get(PASS_KEY) for r in out_rows}) > 1:
        raise DedupError(f"{dest}: cleaned file still mixes evaluation passes")
    original = set(plan["data_lines"])
    for line in out_lines:
        if line not in original and line.rstrip("\n") + "\n" not in original:
            raise DedupError(f"{dest}: emitted a line absent from the original")


def main() -> None:
    args = parse_args()
    if args.apply and not args.out_dir:
        raise SystemExit("--apply requires --out-dir")
    if args.out_dir and os.path.abspath(args.out_dir) == os.path.abspath(args.root):
        raise SystemExit("--out-dir must differ from --root; originals are never modified")

    paths = sorted(
        os.path.join(dirpath, name)
        for dirpath, _, names in os.walk(args.root)
        for name in names if name.endswith(".csv")
    )
    if not paths:
        raise SystemExit(f"no CSVs found under {args.root}")

    plans, failed, clean = [], [], 0
    for path in paths:
        try:
            plan = plan_file(path, args.expected_combos, args.keep)
        except DedupError as e:
            logger.error("skipping %s: %s", os.path.relpath(path, args.root), e)
            failed.append(path)
            continue
        if plan is None:
            clean += 1
        else:
            plans.append(plan)

    logger.info("Scanned %d CSVs under %s", len(paths), args.root)
    logger.info("  already unique : %d", clean)
    logger.info("  with duplicates: %d", len(plans))
    if failed:
        logger.warning("  unreadable     : %d", len(failed))

    for plan in plans:
        rel = os.path.relpath(plan["path"], args.root)
        detail = ", ".join(
            f"initial_acc={v} ({n} rows, {c} combos)" for v, n, c in plan["passes"])
        logger.info("%s: %d -> %d rows | passes: %s | keeping initial_acc=%s%s",
                    rel, plan["n_original"], plan["n_kept"], detail,
                    plan["chosen_pass"], "" if plan["complete"] else "  [INCOMPLETE]")

    if not args.apply:
        logger.info("Audit only. Re-run with --out-dir <dir> --apply to write.")
        return

    written = 0
    for path in paths:
        if path in failed:
            continue
        dest = os.path.join(args.out_dir, os.path.relpath(path, args.root))
        plan = next((p for p in plans if p["path"] == path), None)
        if plan is None:
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            shutil.copy2(path, dest)
        else:
            write_cleaned(plan, dest)
        written += 1

    logger.info("Wrote %d files to %s (%d rewritten, %d copied verbatim)",
                written, args.out_dir, len(plans), written - len(plans))
    if failed:
        logger.warning("%d unreadable file(s) were NOT copied: %s",
                       len(failed), [os.path.relpath(p, args.root) for p in failed])


if __name__ == "__main__":
    main()
