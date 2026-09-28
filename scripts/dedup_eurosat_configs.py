"""Deduplicate EuroSAT training artifacts before SEU evaluation.

A mid-cycle restart of the 63-combo training grid left older (often
near-chance) runs alongside the fixed re-runs. For each
(prior, activation, prior_b) combo this keeps ONE run and archives the rest,
moving the complete per-timestamp artifact set (config, model, guide,
param_store, accuracy/loss CSVs, predictions, training plot) into
``_archive_dupes/`` so nothing is deleted.

Default policy is ``latest`` (newest timestamp) because a restart means the
newer run supersedes the older one; ``best`` (highest best_accuracy) is
available but is unsafe when broken early runs coexist with fixed ones.

Dry-run by default. Pass --apply to actually move files.

Usage:
    # audit one variant dir (no changes)
    uv run --with numpy python scripts/dedup_eurosat_configs.py \\
        --search-dir results/eurosat/bayesian/results_eurosat_v02_00

    # apply keep-latest to one variant dir
    uv run --with numpy python scripts/dedup_eurosat_configs.py \\
        --search-dir results/eurosat/bayesian/results_eurosat_v02_00 --apply
"""
import argparse
import glob
import json
import logging
import os
import re
import shutil
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

TS_RE = re.compile(r"(\d{8}_\d{6})")
ARCHIVE_DIRNAME = "_archive_dupes"
CHANCE_THRESHOLD = 0.20  # 10-class EuroSAT: anything near 0.10 is untrained


def parse_args():
    parser = argparse.ArgumentParser(description="Dedup EuroSAT training runs per combo")
    parser.add_argument("--search-dir", type=str, required=True,
                        help="A single EuroSAT variant dir containing config_*.json")
    parser.add_argument("--keep", type=str, default="latest", choices=["latest", "best"],
                        help="Which run to keep per combo. Default: latest (restart-safe).")
    parser.add_argument("--apply", action="store_true",
                        help="Actually move losing runs to _archive_dupes/. Omit for dry-run.")
    return parser.parse_args()


def combo_key(config: dict) -> tuple:
    """(prior, activation, prior_b) identifies one grid cell."""
    return (config.get("prior"), config.get("activation"),
            config.get("prior_params", {}).get("b"))


def scan(search_dir: str) -> dict:
    """Group config files by combo -> list of run dicts.

    Each run: {ts, acc, act, prior}. act/prior come from the config so the
    archiver can match only that run's files even when June crash-runs share a
    timestamp across different combos.
    """
    groups = defaultdict(list)
    for path in glob.glob(os.path.join(search_dir, "config_*.json")):
        match = TS_RE.search(os.path.basename(path))
        if not match:
            logger.warning("no timestamp in %s, skipping", path)
            continue
        try:
            config = json.load(open(path))
        except (json.JSONDecodeError, OSError) as e:
            logger.error("cannot read %s: %s", path, e)
            continue
        groups[combo_key(config)].append({
            "ts": match.group(1),
            "acc": config.get("best_accuracy"),
            "act": config.get("activation"),
            "prior": config.get("prior"),
        })
    return groups


def pick_keeper(runs: list, keep: str) -> dict:
    """Return the run dict to keep."""
    if keep == "latest":
        return max(runs, key=lambda r: r["ts"])
    return max(runs, key=lambda r: (r["acc"] if r["acc"] is not None else -1.0))


def archive_run(search_dir: str, run: dict, apply: bool) -> int:
    """Move one run's artifact set into _archive_dupes/.

    A run's files all contain BOTH the timestamp and the ``_{act}_{prior}_``
    token, so a timestamp collision across combos cannot drag another combo's
    files along. Returns the number of files moved (or that would be).
    """
    ts, act, prior = run["ts"], run["act"], run["prior"]
    token = f"_{act}_{prior}_"
    archive_dir = os.path.join(search_dir, ARCHIVE_DIRNAME)
    victims = [p for p in glob.glob(os.path.join(search_dir, f"*{ts}*"))
               if os.path.isfile(p) and token in os.path.basename(p)]
    if apply and victims:
        os.makedirs(archive_dir, exist_ok=True)
        for p in victims:
            shutil.move(p, os.path.join(archive_dir, os.path.basename(p)))
    return len(victims)


def main() -> None:
    args = parse_args()
    if os.path.basename(args.search_dir.rstrip("/\\")) == ARCHIVE_DIRNAME:
        raise SystemExit("refusing to run inside an _archive_dupes directory")

    groups = scan(args.search_dir)
    logger.info("Found %d combos in %s", len(groups), args.search_dir)

    dupe_combos = {k: v for k, v in groups.items() if len(v) > 1}
    logger.info("Combos with duplicates: %d", len(dupe_combos))

    # Safety audit: warn about combos whose KEPT run is still near-chance.
    broken = []
    total_moved = 0
    for key, runs in sorted(groups.items(), key=lambda kv: str(kv[0])):
        keeper = pick_keeper(runs, args.keep)
        keep_ts, keep_acc = keeper["ts"], keeper["acc"]
        if keep_acc is not None and keep_acc < CHANCE_THRESHOLD:
            broken.append((key, keep_ts, keep_acc))

        losers = [r for r in runs if r["ts"] != keep_ts]
        if not losers:
            continue
        logger.info("combo %s -> keep %s (acc=%s)", key, keep_ts, keep_acc)
        for loser in losers:
            n = archive_run(args.search_dir, loser, args.apply)
            total_moved += n
            logger.info("    %s %s (acc=%s): %d files",
                        "MOVED" if args.apply else "would move",
                        loser["ts"], loser["acc"], n)

    if broken:
        logger.warning("%d combos KEEP a near-chance (<%.2f) model — inspect before SEU:",
                       len(broken), CHANCE_THRESHOLD)
        for key, ts, acc in broken:
            logger.warning("    %s ts=%s acc=%.4f", key, ts, acc)

    action = "Moved" if args.apply else "Would move"
    logger.info("%s %d files across %d duplicate combos. %s",
                action, total_moved, len(dupe_combos),
                "" if args.apply else "(dry-run — pass --apply to execute)")


if __name__ == "__main__":
    main()
