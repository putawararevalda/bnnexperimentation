# Seeded SEU Script — Precautionary, NOT In Use

**Status: written, imports verified, determinism NOT yet empirically verified.**
**Do not use for paper results unless explicitly decided otherwise.**

Created 2026-08-04.

## What it is

`scripts/eval_seu_eurosat_seeded.py` — a deterministic variant of
`scripts/eval_seu_eurosat.py`.

The default script's Monte-Carlo inference (`S=10` guide samples per batch in
`NewInjector._predict_probs`) is **unseeded**. Re-running the same config
therefore produces slightly different `initial_accuracy`, `accuracy_change`,
and `softmax_difference` values each time. That variance is inherent to MC
sampling — not a bug — but it means published numbers are not exactly
reproducible run-to-run.

The seeded variant pins the RNG before every MC evaluation so a config
reproduces its CSV byte-for-byte on the same machine + library versions.

## Why it is NOT the default

All EuroSAT SEU results collected so far (v02_00, v02_01, v02_02, v02_03) come
from the **unseeded** script. Mixing seeded and unseeded output would make
run-to-run variance inconsistent across the grid, which is worse than having
uniform (if nonzero) variance everywhere.

**Only use it if:**
- a reviewer explicitly asks for exact reproducibility, or
- a specific result needs regenerating byte-for-byte, or
- a future experiment is started from scratch and can use it for *all* configs.

If used, always write to a **separate `--save-dir`** (default is
`results/eurosat/seu_seeded`, deliberately not `results/eurosat/seu`).

## Design notes

- **Nothing in the existing pipeline was modified.** The seeded script
  *imports* `NewInjector`, `load_model`, `load_completed_combos`, and
  `CSV_FIELDS` from `eval_seu_eurosat.py` and subclasses the injector. The
  reference implementation stays the single source of truth.
- **Per-flip seeds are hash-derived**, not sequential:
  `sha256(base_seed | timestamp | location_index | layer | module | bit | param)`.
  This makes each flip's value independent of *evaluation order*, which is what
  keeps it **resume-safe** — a config interrupted at flip 40 and resumed later
  yields the same values as an uninterrupted run. Plain sequential seeding
  (`seed + i`) would break under resume.
- Retains all behaviour of the default script: incremental per-flip CSV writes,
  combo-level resume, variant switch inference, `--fast-smartpool`.

## OUTSTANDING — verification not completed

A verification script exists at **`tests/verify_seeded_determinism.py`** but was
**not run to completion** (it was cancelled to save GPU time; each run loads a
real checkpoint and does several full MC-10 passes over the 5400-image test
set, ~10-20 min).

It checks three things:

| Check | What it proves |
|---|---|
| **A** — two independent seeded runs, same flip order | values reproduce exactly |
| **B** — seeded run with flips in **reversed** order | order-independence → resume is safe |
| **C** — two **unseeded** runs | they *do* differ, so A isn't vacuously passing |

Run it before trusting the script:

```powershell
uv run --with pyro-ppl,torch,torchvision,tqdm,scikit-learn,python-dotenv,requests,pandas,numpy python tests/verify_seeded_determinism.py
```

Expect `PASS / PASS / PASS`. **Check B especially** — if it reports
`ORDER-DEPENDENT`, the resume logic would silently produce inconsistent
values within a single CSV, and the hash-derivation needs revisiting.

What *has* been verified so far: the module imports cleanly, and `combo_seed`
is deterministic for identical inputs while differing across combos.

## Related

- `src/models/components_fast.py` — `SmartPoolFast`, verified bit-identical to
  `SmartPool` (16 case classes + 50-seed fuzz + end-to-end on a real
  checkpoint). Opt in via `--fast-smartpool`. Equivalence test:
  `tests/verify_smartpool_equivalence.py`. That one **was** run and passed.
- MC variance is the reason the seeded script exists, and is also why
  re-running an existing config will never exactly match its stored CSV.
