# ShipsNet k=5 Cross-Validation — Design Spec

**Date:** 2026-07-22
**Status:** Approved, pending implementation plan
**Driver:** ICAART 2026 Paper #90, Reviewer #1 major issue — *"the models are trained only once using an 80/20 train-test split. A single training run is insufficient... employ k-fold cross-validation, using for instance k = 10."*

---

## 1. Goal

Produce mean ± standard deviation across 5 folds for every ShipsNet result the paper's
conclusions rest on — both clean accuracy and the SEU robustness metrics (AAD, Softmax
Difference, ARIn) — so the reported differences can be shown to be genuine rather than
artifacts of one train/test partition.

Secondary outcome: closes Reviewer #1's minor issue *"it would also be interesting to show
the standard deviation to illustrate the variability"* (Table 1).

Scope is ShipsNet only. EuroSAT SEU remains a separate, still-unstarted workstream.

---

## 2. Key finding: the existing split is a valid fold 1

Verified against `datasplit/shipsnet_split_indices.pkl` and the `ImageFolder` targets:

```
dataset:  4000 images  (3000 no_ship / 1000 ship)  = 75/25
train:    3200         (2400 no_ship /  800 ship)  = 75/25
test:      800         ( 600 no_ship /  200 ship)  = 75/25
overlap: 0   duplicates: 0   union == exactly {0..3999}
```

The split is clean and perfectly stratified. It is **not** corrupt.

4000 / 5 = 800 per fold exactly, and a stratified block of 800 requires exactly 600 no_ship
+ 200 ship — which is precisely the existing test set. The remaining 3200 train indices
(2400/800) partition into 4 further blocks of exactly 600/200, with no remainder.

In 5-fold CV, fold 1's train set is the union of the other four blocks = the existing 3200
train indices. Therefore **the 252 already-trained ShipsNet models are fold 1**, bit-for-bit.
They are relabelled, not retrained. Only folds 2–5 are new work.

This reuse is the reason the project is affordable, and it is the reason we do **not** use
`sklearn.StratifiedKFold` to generate folds from scratch (see §4, Rejected Approaches).

---

## 3. Decisions

| Decision | Choice | Rationale |
|---|---|---|
| CV scope — training | **Full grid**: 7 activations × 3 priors × 3 b-values × 4 variants = 252 configs | Maximum coverage; supports std on every table cell |
| CV scope — SEU | **Full grid, all 5 folds** | Puts mean±std on AAD / SoftmaxDiff / ARIn themselves, directly answering the reviewer |
| What varies per fold | **Split only.** `torch.manual_seed(42)` fixed across all folds | Textbook k-fold. Unconfounded claim: "not an artifact of a particular partition." Also keeps fold 1 identical to existing runs, preserving reuse |
| k | **5**, not the reviewer's suggested 10 | Reviewer wrote "for instance k = 10"; k=5 is a standard, defensible choice. k=10 would double an already ~10-day budget |

---

## 4. Approach

**Chosen:** a single fold-manifest pickle, plus a `--fold` flag threaded through the existing
training and SEU scripts. Fold 1 is pinned to the existing split by construction.

**Rejected — compute folds on the fly with `sklearn.StratifiedKFold`:** would generate a fold 1
that does not match the existing split, discarding 252 trained models (~1.4 days of GPU) and
84 completed SEU sweeps. The single largest cost saving in this plan comes from that reuse.

**Rejected — five separate split pickles:** functionally equivalent but spreads an invariant
(the five blocks must be mutually disjoint and jointly exhaustive) across five files where it
cannot be checked atomically. One manifest keeps the invariant verifiable in one place.

---

## 5. Components

### 5.1 `scripts/make_shipsnet_folds.py` — new

Generates `datasplit/shipsnet_folds_k5.pkl`.

- Fold 1 test block := existing `shipsnet_split_indices.pkl['test']`, **verbatim**.
- Remaining 3200 indices → 4 stratified blocks of exactly 600 no_ship + 200 ship,
  partitioned deterministically with `numpy.random.default_rng(42)` shuffling each class's
  index list before it is chunked. The seed is recorded in the manifest so the partition is
  reproducible from the manifest alone.
- Each fold's `train` := union of the other four blocks (3200); `test` := its own block (800).

Structure:

```python
{
  "k": 5,
  "folds": [{"train": [...3200], "test": [...800]}, ...],   # 5 entries, fold N at index N-1
  "meta": {"seed": 42, "source_split": "datasplit/shipsnet_split_indices.pkl",
           "created": "2026-07-22", "class_ratio": {"no_ship": 3000, "ship": 1000}}
}
```

**Assertions — the script must fail loudly rather than emit a bad manifest:**

1. 5 test blocks, each of length exactly 800
2. Test blocks pairwise disjoint
3. Union of test blocks == `set(range(4000))`
4. Each test block has exactly 600 no_ship + 200 ship
5. For each fold: `train ∩ test == ∅` and `len(train) == 3200`
6. `folds[0]["test"] == legacy["test"]` and `folds[0]["train"] == legacy["train"]` (as sets)

Assertion 6 is the load-bearing one. If it fails, every claim of fold-1 reuse is void and
nothing downstream may run.

### 5.2 `src/data/shipsnet.py` — modify

```python
def load_data(batch_size: int = 16, fold: int | None = None):
```

- `fold=None` → current code path, byte-identical. Nothing existing breaks.
- `fold in 1..5` → reads `datasplit/shipsnet_folds_k5.pkl`, uses `folds[fold-1]`.
- `torch.manual_seed(42)` stays fixed regardless of fold, per §3.
- Invalid fold → `ValueError`.

`load_data_withval` gets the same parameter for signature consistency.

### 5.3 `scripts/train_shipsnet.py` — modify

- Add `--fold N` (default `None` → legacy behaviour).
- Pass `fold` into `load_data`.
- Write `"fold": N` into the config JSON.
- Also write the **variant** fields into the config JSON (see §6.1).
- Save to `results/shipsnet/bayesian/fold{N}/results_shipsnet_v02_0{X}/`.
  Fold 1 artifacts stay where they are — no files are moved.

### 5.4 `scripts/backfill_config_metadata.py` — new

Stamps the 252 existing fold-1 config JSONs with:

- `"fold": 1`
- the variant fields, derived from the containing directory name

Idempotent — safe to re-run. Writes nothing if the keys are already correct.

### 5.5 `scripts/eval_seu_shipsnet.py` — modify

- Add `--fold N`; **pass it into `load_data(fold=N)`**.
- Propagate `fold` and the variant fields into every output CSV row.
- Save to a per-fold directory.

> **Correctness hazard.** If the SEU script loads a fold-3 model but evaluates it against the
> legacy test set, it scores the model on images it was trained on. Every resulting number
> would be silently wrong — no crash, no warning, just optimistic garbage. The fold argument
> reaching `load_data` is the single most important line in this change set, and the pilot
> (§7 step 4) exists largely to confirm it.

### 5.6 `scripts/aggregate_seu.py` — new

**No aggregation script exists anywhere in the repository.** The paper's current tables were
produced outside version control. Without this component the project ends with ~1,400 CSVs
and no tables.

- Recursively scan SEU output directories; read `fold` and variant from each row.
- Compute per-row `AAD = |accuracy_change|` and
  `ARIn = sqrt((AAD² + SoftmaxDiff²) / 2)` via `src/evaluation/metrics.py`.
- Group by `(variant, activation, prior, b, layer, module, bit)` and reduce **across folds**
  to mean ± std, reporting `n_folds` per cell.
- Emit CSV and LaTeX. LaTeX output must respect ICAART column widths — Reviewer #1 flagged
  Tables 3, 4 and 5 as out of page bounds.
- Rows where `remarks` marks a degenerate injection (negative scale/width) carry `NaN`;
  these must be excluded from mean/std with the excluded count reported, not silently dropped.

---

## 6. Two schema findings

### 6.1 Model variant is not recorded in the config JSON

Current config JSON:

```json
{"activation": "actRWG", "prior": "gaussian", "num_epochs": 100,
 "best_accuracy_at_epoch": 90, "best_accuracy": 0.6453125,
 "batch_size": 16, "train_size": 3200,
 "prior_params": {"mu": 0.0, "b": 10.0}}
```

There is no `smartpool` / `dropout` / `weight_decay` field — the variant is encoded only in the
directory name (`results_shipsnet_v02_00..03`). Aggregation cannot group by variant reliably
from directory strings alone once a fold layer is added to the path.

**Resolution:** training writes explicit variant fields going forward; the backfill script
(§5.4) adds them to the 252 existing configs, derived from their directory.

### 6.2 The last-epoch / best-checkpoint inconsistency resolves itself — for free

`CLAUDE.md` records a pending TODO: `train_shipsnet.py` logs `test_acc` from the **last-epoch**
model, while the SEU scripts load the `*_epoch_best_*` artifacts. Averaging last-epoch fold 1
against best-checkpoint folds 2–5 would inflate the standard deviation with a bookkeeping
artifact rather than real fold variance — an indefensible number to put in front of a reviewer
who already scored technical quality 1/6.

However, the SEU CSV already contains an `initial_accuracy` column, and `NewInjector` computes
it from the **best checkpoint** with MC-10 sampling, on that fold's test set. Since the full
grid × all folds SEU sweep is in scope (§3), *every* config in *every* fold will have a
best-checkpoint, MC-10, correct-test-set accuracy recorded uniformly.

**Resolution:** source the accuracy table from the SEU CSVs' `initial_accuracy` column, not from
the training-time `test_acc`. This is consistent by construction, costs nothing, and closes the
pending TODO for ShipsNet. The separately-planned 2-hour re-evaluation pass is **not needed**.

The training-side `test_acc` remains last-epoch and should simply not be used for paper tables.

---

## 7. Execution order

| # | Step | Est. |
|---|---|---|
| 1 | Generate folds; assert fold 1 == legacy split | minutes |
| 2 | Backfill `fold` + variant into 252 existing configs | seconds |
| 3 | **Pilot**: one config through folds 2–5, train → SEU → aggregate | ~1 h |
| 4 | Training sweep, folds 2–5, full grid (1008 runs @ ~8 min) | ~5.6 days |
| 5 | SEU sweep (1008 new fold 2–5 + 168 fold-1 b≠1.0 = 1176 @ ~5 min) | ~4 days |
| 6 | Final aggregation, emit paper tables | ~1 h |

**Total ≈ 10 days of continuous GPU**, with the EuroSAT SEU sweep still queued behind it.

Step 3 is non-negotiable. It must confirm, on a single config, that: fold data actually differs
per fold, the SEU script evaluates against the right test set, variant/fold survive into the
CSVs, and the aggregation groups correctly. Discovering a grouping or naming bug after step 5
would cost the full nine days.

Steps 4 and 5 are resumable — the SEU script already skips timestamps whose CSV exists, and
training is per-config idempotent by timestamp.

---

## 8. Timing basis

Measured from artifact mtimes in the existing results tree, not estimated:

- Training: median 6 min, mean ~8 min per config (100 epochs)
- SEU: median ~5 min per model sweep (168 injections × MC-10 over the test set)

Existing SEU coverage is **b=1.0 only** — 21–22 CSVs per variant directory, i.e. the
`--limited-mode` subset. Extending SEU to the full grid therefore adds 168 fold-1 sweeps on top
of the fold 2–5 work.

---

## 9. Out of scope

- EuroSAT k-fold and the EuroSAT SEU sweep (separate workstream, SEU not yet started)
- k=10 (see §3)
- Deterministic-baseline CV — `eval_seu_deterministic.py` is untouched; extend only if the
  DNN-vs-BNN comparison in Table 2 also needs error bars
- The ARIn justification itself (Reviewer #1's separate major issue) — this spec supplies the
  variance data that discussion will draw on, but the argument is a writing task

---

## 10. Success criteria

1. `make_shipsnet_folds.py` passes all six assertions, including `fold1 == legacy`
2. Pilot confirms per-fold test sets differ and SEU evaluates against the correct one
3. All 252 configs have results for 5 folds
4. `aggregate_seu.py` emits mean ± std tables with `n_folds` per cell and an explicit
   count of excluded degenerate injections
5. Accuracy figures trace to SEU `initial_accuracy` (best checkpoint, MC-10) throughout —
   no last-epoch values in any paper table
