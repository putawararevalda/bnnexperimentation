# SEU metric noise floors — MC sampling noise (BNN) and a double-softmax bug (DNN)

> Generated: 2026-08-10
> Data: `results/shipsnet/seu/results_shipsnet_v02_00_SEU/` (BNN, fold 1),
> `results/eurosat/seu_clean/v02_0{0,1,2,3}/` (BNN, `b=1.0` slices),
> `results/shipsnet/deterministic/results_shipsnet_deterministic_00_SEU/` (DNN).
> Reproduce: `uv run --with pandas,numpy python scripts/analyze_seu_noise_floor.py`
> Paper: `docs/2026_ICAART_Revalda (14).pdf`

## Summary

A large fraction of the reported `softmax_difference` (ASD) — and therefore of
ARIn — does not measure SEU damage. Two **separate and unrelated** causes:

| | Cause | Affects | Share of ASD | Status |
|---|---|---|---|---|
| **1** | MC-10 sampling noise | BNN, both datasets | **49–58%** | Inherent to the method |
| **2** | Double-softmax bug | DNN only | **97%** | **Code defect** |

Finding 1 is a measurement-precision issue that affects ShipsNet and EuroSAT
roughly equally, so BNN-vs-BNN comparisons survive. **Finding 2 is a bug that
inflates the DNN side only, and it undermines the paper's headline
BNN-vs-DNN claim.**

---

## Method: the null channel

`initial_accuracy` is computed **once per config** and reused as the baseline for
all ~168 injections of that config, so its error enters every `accuracy_change`
as a systematic offset rather than averaging away. `softmax_difference` is worse:
it compares two independently drawn MC-10 passes, so even a completely inert flip
returns a non-zero value.

Mantissa bits provide a null channel. Flipping bit 21 perturbs a weight by
~1.2e-4 relative; bit 10 by ~2.5e-1 — a **2,000× range**. If the metric tracked
real damage, they would differ substantially.

Two diagnostics separate noise from signal:
- **Agreement across mantissa bits of very different magnitude** → a floor.
- **`signal_ratio` = |mean signed change| / mean |change|**: ~0 means symmetric
  scatter (noise); ~1 means systematic damage (signal).

---

## Finding 1 — BNN: MC-10 sampling noise

### ShipsNet fold 1 (v02_00, the paper run), 3,276 valid injections

| bit | rel. magnitude | AAAD | ASD | mean signed | signal_ratio |
|---|---|---|---|---|---|
| 0 (sign) | — | .00678 | .05527 | −.00134 | 0.20 |
| **1 (exp MSB)** | — | **.12327** | **.41962** | −.12064 | **0.98** |
| 3 (exp) | — | .00637 | .05424 | −.00093 | 0.15 |
| 6 (exp) | — | .00635 | .05443 | −.00057 | 0.09 |
| 10 (mant) | ~2.5e-1 | .00608 | .05457 | −.00112 | 0.18 |
| 15 (mant) | ~7.8e-3 | .00616 | .05406 | −.00103 | 0.17 |
| 21 (mant) | ~1.2e-4 | .00611 | .05414 | −.00100 | 0.16 |

Mantissa bits spanning a 2,000× magnitude range all return **AAAD ≈ .0061** and
**ASD ≈ .0543**. That constancy is the floor.

**On ShipsNet, only bit 1 rises above it.** Bits 0, 3 and 6 sit at .0064–.0068
with `signal_ratio` 0.09–0.20 — statistically indistinguishable from the inert
mantissa bits. The earlier observation in
`eurosat_v00_vs_shipsnet_fold1_seu.md` that "on ShipsNet, bit 0 is
indistinguishable from the mantissa floor" now has its mechanism: it *is* the floor.

### Floors across all slices

| Slice | AAAD | ASD | ARIn | floor AAAD | floor ASD | ASD floor % | ARIn (floor-removed) |
|---|---|---|---|---|---|---|---|
| ShipsNet fold 1 | .02427 | .11057 | .08004 | .00612 | .05426 | **49%** | .04183 |
| EuroSAT v02_00 base | .04263 | .12713 | .09481 | .00419 | .07340 | **58%** | .04671 |
| EuroSAT v02_01 smartpool | .04268 | .12806 | .09545 | .00377 | .07414 | **58%** | .04702 |
| EuroSAT v02_02 dropout | .04252 | .12425 | .09286 | .00440 | .06755 | **54%** | .04831 |
| EuroSAT v02_03 weight_decay | .04250 | .12663 | .09445 | .00427 | .07319 | **58%** | .04646 |

The floor is stable across all four EuroSAT variants, confirming it is a property
of the measurement rather than of any model variant.

> **Floor-removed columns are a sensitivity check, not corrected metrics.** ASD is
> an L∞ softmax distance; its noise does not subtract linearly. Direction and
> rough magnitude only.

### Effect on the ShipsNet-vs-EuroSAT comparison

| Claim | Headline | Floor-removed |
|---|---|---|
| EuroSAT less robust (ARIn) | +19% | +12% |
| EuroSAT worse AAAD | +75% | +112% |
| EuroSAT worse ASD | +15% | **−5%** |

The ARIn conclusion holds with reduced magnitude. The **ASD conclusion reverses**:
the apparent 15% confidence-stability gap was mostly the two datasets having
different noise floors (49% vs 58%).

This *strengthens* the paper's mechanism argument. With the floor removed, ASD is
essentially flat across datasets while AAAD roughly doubles — a cleaner statement
of "BNN robustness is confidence stability, not label stability".

---

## Finding 2 — DNN: double-softmax bug

### Evidence

The deterministic CNN has no MC sampling, so its floor should be exactly zero.
Its **AAAD floor is** exactly that: `.000000` at bits 15 and 21. But its **ASD
floor is .26065** — on flips that provably changed nothing.

| bit | AAAD | ASD | mean signed |
|---|---|---|---|
| 1 | .09955 | .32152 | −.09884 |
| 15 | **.000000** | **.26065** | .000000 |
| 21 | **.000000** | **.26065** | .000000 |

DNN headline: AAAD `.01453`, ASD `.26943`, ARIn `.19079`. **97% of that ASD is
present on inert flips.**

### Root cause

`scripts/eval_seu_deterministic.py:82` defines:

```python
def _softmax_diff(self, before_probs, after_logits, penalty=1.0):
    ...
    safe[finite] = F.softmax(after_t[finite], dim=1)   # applies softmax
```

`_predict_probs` (line 75) returns `all_probs` **already softmaxed**. Line 116
then passes those probabilities into the `after_logits` slot:

```python
softmax_diff = self._softmax_diff(self.initial_probs, after_probs)   # BUG
```

so softmax is applied **twice** to the "after" side and once to "before". The
metric compares `p` against `softmax(p)`.

The BNN scripts do **not** have this bug — both pass logits correctly:
- `scripts/eval_seu_shipsnet.py:196` → `self._softmax_diff(self.initial_probs, after_logits)`
- `scripts/eval_seu_eurosat.py:220` → `self._softmax_diff(self.initial_probs, after_logits)`

### Numerical confirmation

For a binary classifier, `L∞|p − softmax(p)|` is:

| p(confident class) | 1.00 | 0.99 | 0.95 | 0.90 | 0.80 | 0.50 |
|---|---|---|---|---|---|---|
| artifact | .2689 | .2629 | .2391 | .2100 | .1543 | .0000 |

The observed floor `.26065` lands precisely in the p≈0.99 band — exactly what a
confident, well-trained binary CNN produces. Mechanism confirmed.

### Impact on the headline claim

The paper's abstract reports **BNN ARIn .0779 vs DNN .1909**. The DNN figure is
dominated by the artifact. Removing the floor crudely gives DNN ASD ≈ `.0088` and
**ARIn ≈ .0120**, versus the BNN's floor-removed `.0418`.

**That reverses the ordering.** The one-line bug is doing most of the work in the
paper's central quantitative claim.

> **Do not quote `.0120`.** The exact corrected value requires re-running the DNN
> SEU grid with the fix; the artifact is not a clean additive constant once a flip
> materially changes the output. What is solid: the DNN ASD column is invalid, and
> the true DNN ARIn is far below `.1909`.

---

## Claim candidates

- **Claim:** ~50% of the BNN ASD is MC-10 sampling noise, stable across datasets and variants.
  - Source evidence: mantissa-bit null channel, 5 slices, 3,276 injections each.
  - Allowed wording: "a measurement floor of ≈.054 (ShipsNet) / ≈.073 (EuroSAT) ASD".
  - Forbidden: any claim that floor-removed values are corrected metrics.
  - Uncertainty: no repeated-run measurement yet (see Blockers).
  - Decision: **keep**.

- **Claim:** On ShipsNet, only bit 1 exceeds the noise floor.
  - Source evidence: bits 0/3/6 at .0064–.0068 vs mantissa .0061, signal_ratio ≤0.20.
  - Allowed: "bit-position effects other than the exponent MSB are not resolvable on ShipsNet".
  - Forbidden: "sign-bit flips cause a moderate accuracy drop" (ShipsNet cannot support this; EuroSAT can, signal_ratio 0.76).
  - Decision: **keep**.

- **Claim:** The DNN `softmax_difference` column is invalid.
  - Source evidence: code defect at `eval_seu_deterministic.py:116` + exact numerical match.
  - Allowed: "the DNN ASD is dominated by a double-softmax artifact and must be regenerated".
  - Forbidden: quoting any corrected DNN ARIn before a rerun.
  - Decision: **keep — blocking for the paper**.

- **Claim:** BNNs are ~2.4× more robust than DNNs (abstract).
  - Decision: **discard until the DNN grid is regenerated.** Currently unsupported.

---

## Blockers and limitations

1. **No repeated-run measurement of MC noise.** The floor is inferred from the
   null channel — strong and self-consistent, but indirect. A direct measurement
   (re-running one config's MC-10 evaluation N times) was prepared but not run.
   Expected to corroborate; a smoke test on one ShipsNet config gave **±0.80 pp**
   std on `initial_accuracy` across 2 draws, consistent with the floor's scale.
2. **n = 1 run per config, unseeded.** No seeds or repeats anywhere in the
   pipeline (verified: neither SEU script nor the training path seeds).
3. **Floor removal is linear subtraction.** Not rigorous for an L∞ metric.
4. **The DNN correction cannot be computed from existing CSVs** — per-sample
   probabilities were not stored. Requires a rerun.
5. **EuroSAT has no DNN SEU grid at all**, so Finding 2 cannot be cross-checked
   on the second dataset.

---

## Recommended actions

| Priority | Action | Cost |
|---|---|---|
| **P0** | Fix `eval_seu_deterministic.py:116` → pass `after_logits`. Rerun the DNN SEU grid (deterministic, no MC — cheap). Every BNN-vs-DNN number in the paper depends on it. | Low |
| **P1** | Report the BNN noise floor in the paper. Costs no compute — the numbers are in this doc — and preempts a reviewer noticing that "negligible" mantissa bits sit well above zero. | None |
| **P2** | Scope bit-level claims to what each dataset resolves: ShipsNet supports bit 1 only; EuroSAT additionally supports bits 0, 3, 6. | None |
| **P3** | Do **not** rerun the EuroSAT BNN SEU grid. Seeding would not reduce the floor — `eval_seu_eurosat_seeded.py` seeds each flip independently of the baseline pass, so the before/after floor persists unchanged. | — |
| **P4** | *(post-submission)* Remove the floor properly via **common random numbers** — use the same weight samples for baseline and post-flip passes, so an inert flip yields near-identical predictions. This changes every BNN number and is not a quick fix. | High |

---

## Relationship to other docs

- Supersedes nothing. Adds a measurement-validity layer beneath
  `eurosat_v00_vs_shipsnet_fold1_seu.md` and `eurosat_seu_clean_vs_paper.md`.
- Those two documents' **BNN-vs-BNN** conclusions remain valid: the floor affects
  both datasets similarly, except for the ASD-gap reversal noted above.
- Their repeated caveat that "no EuroSAT deterministic-CNN SEU runs exist" is now
  more consequential — the ShipsNet DNN grid, the only one that exists, needs
  regenerating.
