# EuroSAT BNN Training — Interactive HTML Explainer

**Date:** 2026-08-10
**Status:** Approved (not committed)

## Goal

A self-contained HTML page that teaches how the EuroSAT Bayesian CNN training
pipeline in this repository actually works — written against the real code, so
that re-reading it months later restores working knowledge of concepts like
MC-10 inference, the guide, the ELBO, and the prior scale `b`.

Primary purpose is **teaching concepts**, not serving as a CLI flag reference.
Flags appear only where they illuminate a concept.

## Scope

**In scope** — the EuroSAT Bayesian training path:

| Concern | File |
|---|---|
| Entry point, sweep, variants | `scripts/train_eurosat.py` |
| Data loading and split | `src/data/eurosat.py` |
| Model and priors | `src/models/bayesian_cnn.py` |
| Guides | `src/utils/guide.py` (`AutoLaplace`, `AutoUniform`) |
| Training loop, checkpointing, MC inference | `src/training/svi.py` |

The page ends at the MC-10 test evaluation and the artifacts written to
`results/eurosat/bayesian/results_eurosat_v02_{variant}/`.

**Out of scope** — SEU/bitflip injection, robustness metrics (AAD, softmax
difference, ARIn), ShipsNet, the deterministic CNN baseline, MLflow. Each may
become its own page later; this spec covers one page only.

## Deliverable

A single file: `docs/learn/eurosat-bnn-training.html`

- Self-contained: inline `<style>` and `<script>`, no external requests, no
  build step, no package dependencies.
- Opens by double-click from the filesystem and works offline.
- Demos use `<canvas>` and vanilla JS. No plotting library.
- Committed to git alongside the code it documents (commit is a separate,
  later decision — this spec is not committed as part of its own creation).

## Teaching spine

The page follows one concrete run end to end:

```
python scripts/train_eurosat.py --variant 00 --prior Gaussian_prior \
    --activation relu --epoch 100 --b-value 1.0
```

Every abstract idea is grounded in that run's verified numbers:

| Quantity | Value | Source |
|---|---|---|
| EuroSAT images | 27,000 | `datasplit/split_indices_v2.pkl` (verified) |
| Train / test | 21,600 / 5,400, zero overlap | same (verified) |
| Batch size | 54 | `load_data(batch_size=54)` |
| Batches per epoch | 400 | 21,600 / 54 |
| `obs_scale` when `--scale-likelihood` | 400.0 | 21,600 / 54 |
| Latent weights | 183,242 | 864+32 + 18,432+64 + 163,840+10 |
| Variational parameters (AutoNormal) | 366,484 | one `loc` + one `scale` each |
| MC samples at inference | S = 10 | `predict_data(..., num_samples=10)` |
| Sweep size per variant | 63 | 3 priors x 7 activations x 3 `b` values |

## Content outline

Nine sections, sticky left navigation.

1. **Why Bayesian at all** — point weights vs. distributions over weights; what
   uncertainty buys for the SEU robustness question this project studies.
2. **The model** — `BayesShipsCNN` layer by layer with a shapes diagram:
   `3x64x64 -> conv1 -> 32x64x64 -> pool -> 32x32x32 -> conv2 -> 64x32x32 ->
   pool -> 64x16x16 -> flatten 16,384 -> fc1 -> 10`. What `PyroSample` does:
   every weight is a fresh draw on every forward pass.
3. **The prior** — gaussian / laplace / uniform; what `b` in {10, 1, 0.1} means
   for each. **Demo 1.**
4. **The guide** — the approximate posterior. `AutoNormal` learns a `loc` and a
   `scale` per weight: 183,242 -> 366,484 learned numbers. `init_scale = 0.25*b`
   (default) and what a poor initialisation does.
5. **The ELBO** — intuition first (fit the data, stay near the prior), then the
   two-term expression. **Demo 2.**
6. **The SVI loop** — what one `svi.step()` does per batch. Accuracy measured at
   epoch 1, every 10th epoch, and the final epoch.
7. **Checkpointing** — "best" is best *training* accuracy; three artifacts
   (model `state_dict`, guide `state_dict`, Pyro param store `.pkl`) and why the
   param store is the essential one to restore (the guide is traced through the
   global store).
8. **MC-10 inference** — `predict_data`: draw S=10 weight sets, run 10 forward
   passes, **average the logits, then argmax**. Why averaging logits differs
   from averaging softmax outputs, and why the result varies run to run.
   **Demo 3.**
9. **Knobs and artifacts** — variants 00-03 (base / smartpool / dropout /
   weight decay), the prior x activation x `b` grid, and the exact filenames a
   run drops into its save directory.

## Caveats

Caveats about the current implementation are included **inline**, at the point
of the concept they affect, in a visually distinct callout style so they never
read as "this is the recommended method."

| Section | Caveat |
|---|---|
| 5 (ELBO) | `--scale-likelihood` defaults off, leaving the KL term over-weighted relative to a correct minibatch ELBO. EuroSAT v02 was trained with it **on** (`obs_scale=400`); ShipsNet folds 1-2 were trained **off**. The two datasets therefore optimise different objectives. |
| 6 (SVI loop) | The `svi.evaluate_loss` inf/nan pre-check `continue`s past the offending batch, so that batch contributes no gradient — silently dropped, not retried. |
| 6 (SVI loop) | Train accuracy is only evaluated at epoch 1, every 10th epoch, and the last, so "best epoch" is chosen from at most ~11 measurements out of 100. |
| 7 (Checkpointing) | Best checkpoint is selected on **training** accuracy — there is no validation split in this path — so an overfit epoch can be banked as "best". |
| 8 (MC-10) | MC sampling is unseeded, so re-running evaluation on the same checkpoint gives slightly different accuracy each time. |

## Demos

Three, all canvas + vanilla JS, roughly 100 lines each.

**Demo 1 — prior shapes.** One canvas over x in [-3, 3]. Radio for family
(gaussian / laplace / uniform), slider for `b` snapped to {10, 1, 0.1}. Draws
the density and annotates that at equal `b`, Laplace has a sharper peak and
heavier tails than Gaussian, and that uniform is hard-bounded at +/- b. Caption
ties it to the run: this is the distribution each of the 183,242 weights is
drawn from before any data is seen.

**Demo 2 — ELBO tradeoff.** Two stacked bars (data term, KL term) and one
slider representing how far the posterior has moved from the prior. Moving
right improves the data term and grows the KL. A toggle switches between
`obs_scale = 1` (legacy) and `obs_scale = 400` (`--scale-likelihood`),
rescaling the data bar so the KL over-weighting is visible. **Illustrative, not
computed from a real run — labelled as such on the page.**

**Demo 3 — MC-10.** Ten logit vectors over the 10 EuroSAT classes, generated
from a seeded JS PRNG around a plausible posterior. "Draw sample" steps one at
a time, showing that sample's argmax alongside the running mean's argmax; the
prediction typically flips early and settles as S grows. "Reroll" regenerates
the set, demonstrating the non-determinism caveat directly.

## Visual design

- Dark-first palette, single accent colour, generous whitespace.
- Monospace for anything quoted from the codebase.
- Code excerpts are verbatim from the source, each labelled `path:line`.
- Sticky left nav on desktop; collapses to a top bar below ~900px.
- `prefers-reduced-motion` respected; no autoplaying animation.

## Correctness and verification

The main risk is quiet drift from the code, so:

- Every numeric claim traces to a file that was read, or to a value computed
  and checked during the build (the split counts above were verified by loading
  the pickle, not assumed from the 80/20 ratio).
- Every code excerpt is copy-pasted from the file with a `path:line` label.
- After building: open in Chrome, exercise all three demos, check desktop and
  narrow viewports, confirm a clean console.

## Non-goals

- No Python execution, no live training, no server.
- Not a replacement for `CLAUDE.md` as the operational reference.
- Does not document ShipsNet, SEU evaluation, or the deterministic baseline.
