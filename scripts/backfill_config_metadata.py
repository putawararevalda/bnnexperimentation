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
