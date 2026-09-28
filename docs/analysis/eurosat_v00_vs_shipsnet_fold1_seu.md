# EuroSAT v00 vs ShipsNet fold 1 — SEU Robustness Comparison

> Generated: 2026-07-31
> Sources: `results/eurosat/seu/v02_00/` (63 CSVs) and `results/shipsnet/seu/results_shipsnet_v02_00_SEU/` (21 CSVs)
> Paper reference: `docs/2026_ICAART_Revalda (14).pdf`
> Reproduce: `uv run --with pandas,numpy python scripts/compare_seu_datasets.py`

## Scope and validation

Both slices are **model variant 0 (base)** only, and both cover the full grid of
7 activations × 3 priors × (3 layers × 2 params × 2 positions × 7 bits, minus the
invalid bit-0 scale attacks) = **3,276 valid injections each**.

Two alignment points:

1. **Prior scale.** EuroSAT v00 swept `prior_b ∈ {0.1, 1.0, 10.0}` (9,828 valid rows);
   ShipsNet fold 1 only ran `b = 1.0`. Unless stated otherwise, every comparison below
   uses the **`b = 1.0`** EuroSAT slice so the grids match exactly.
2. **Paper agreement.** Recomputing ShipsNet fold-1 variant-0 from the raw CSVs gives
   AAAD `.0243` / ASD `.1106` / ARIn `.0800`, which reproduces the paper's Table 3 row 0
   (BNN) exactly. This confirms fold 1 is the paper run and that the aggregation here
   matches the one used for the paper tables.

Metrics follow the paper: `AAAD = mean|acc_after − acc_before|`, `ASD = mean softmax difference`,
and `ARIn = sqrt((AAAD² + ASD²)/2)` computed from the **aggregated** AAAD and ASD.
`initial_accuracy` comes from the SEU CSVs (best-checkpoint models) on both sides, so the
last-epoch/best-checkpoint discrepancy noted in `CLAUDE.md` does not affect this comparison.

## 1. Headline

| Slice | init acc | AAAD | ASD | ARIn |
|---|---|---|---|---|
| ShipsNet fold 1, v00 (paper) | .8449 | .0243 | .1106 | **.0800** |
| EuroSAT v00, `b=1.0` | .7953 | .0426 | .1271 | **.0948** |
| EuroSAT v00, all `b` | .7618 | .0422 | .1267 | .0944 |

EuroSAT is **~19% less robust** by ARIn. The gap is almost entirely in accuracy, not in
prediction confidence: AAAD rises 75% (.0243 → .0426) while ASD rises only 15%
(.1106 → .1271).

**Interpretation.** This is the expected multiclass effect. With 10 classes, a perturbed
logit vector has nine ways to cross a decision boundary instead of one, so a comparable
softmax shift converts into far more label flips. The Bayesian confidence-stability
argument in the paper (Section 4.2.3 — MC averaging dilutes logit shifts) survives the
move to EuroSAT; the accuracy-side conclusion weakens.

### EuroSAT-only: prior scale as a robustness variable

| `prior_b` | AAAD | ASD | ARIn |
|---|---|---|---|
| 0.1 | .0377 | .1203 | **.0891** |
| 1.0 | .0426 | .1271 | .0948 |
| 10.0 | .0462 | .1326 | .0993 |

Robustness degrades monotonically as the prior widens. There is no ShipsNet v00
counterpart (fold 1 ran `b=1.0` only), so this is currently a single-dataset finding.

## 2. What replicates

All of the paper's dataset-invariant claims hold on EuroSAT.

| Finding | ShipsNet f1 (ARIn) | EuroSAT v00 (ARIn) |
|---|---|---|
| Bit 1 (exponent MSB) is catastrophic | .3093 vs ~.039 elsewhere | .3331 vs ~.052 elsewhere |
| 0→1 flips far worse than 1→0 | .1315 vs .0386 | .1336 vs .0540 |
| fc1 is the least robust layer | fc1 bias .1277 / weight .1238 | fc1 bias .1388 / weight .1120 |
| Scale/spread worse than center | Gaussian .0982 (S) vs .0621 (C) | Gaussian .1050 (S) vs .0909 (C) |
| Mantissa bits (10/15/21) negligible | .0385–.0388 | .0519–.0520 |
| ReLU is the least robust activation | .1239 (worst of 7) | .1332 (worst of 7) |
| RWG best AAAD | .0160 (best) | .0257 (best) |

## 3. What does not replicate — three conclusions flip

### 3.1 Prior ranking inverts: Uniform beats Gaussian on EuroSAT

| Prior | ShipsNet AAAD / ASD / ARIn | EuroSAT AAAD / ASD / ARIn |
|---|---|---|
| gaussian | .0218 / .1092 / **.0787** | .0406 / .1316 / .0974 |
| laplace | .0244 / .1102 / .0798 | .0410 / .1384 / .1021 |
| uniform | .0266 / .1123 / .0816 | .0462 / .1114 / **.0853** |

The driver is the ASD term. On EuroSAT, Uniform has the *worst* AAAD (.0462) but the
*best* ASD (.1114), and ASD dominates ARIn. The paper's own Section 4.4.3 mechanism —
a flat PDF means a scale perturbation changes sampling probability less than it does
for a peaked Gaussian/Laplace — predicts exactly this; it simply was not strong enough
to dominate ARIn in the binary ShipsNet setting.

Split by center vs scale, EuroSAT shows Uniform winning on **both** parameter types
for ASD (C `.1027`, S `.1216`), while Gaussian/Laplace sit at `.124–.146`.

**Paper impact.** The abstract's "Gaussian priors outperforming Laplace and Uniform"
is ShipsNet-specific as written.

### 3.2 Sigmoid is no longer the robustness winner

| Activation | ShipsNet ARIn | EuroSAT ARIn | EuroSAT init acc |
|---|---|---|---|
| actRWG | .0650 | **.0658** | .82–.83 |
| actWG | .0681 | .0687 | .83–.86 |
| sin | .0809 | .0713 | .81–.85 |
| tanh | .0813 | .0863 | .79–.83 |
| sigmoid | **.0647** | .1169 | **.63–.67** |
| relu6 | .0767 | .1237 | .77–.81 |
| relu | .1239 | .1332 | .80–.81 |

Sigmoid drops from best (ShipsNet) to 5th of 7 (EuroSAT) *and* collapses in accuracy —
.63–.67 versus .79–.86 for every other activation. The paper's small-gradient argument
(Section 4.5: h′ ≈ 0.25 near zero) stops being a free win once the network has to
separate 10 classes; the saturation that damps perturbations also damps learning.

On EuroSAT the robustness leaders are the Dennis & Pope activations (RWG, WG) plus sin,
which is a cleaner story than sigmoid because they do not pay an accuracy penalty.

**Paper impact.** "Sigmoid was the most robust activation despite an accuracy trade-off"
does not generalize. On EuroSAT the trade-off becomes prohibitive rather than acceptable.

### 3.3 relu6 loses its advantage

ShipsNet ARIn `.0767` (close to sigmoid, clearly better than relu `.1239`); EuroSAT
ARIn `.1237`, essentially indistinguishable from plain relu (`.1332`). Clipping at 6
stops helping when the pre-activation range needed for 10-class discrimination is larger.

## 4. Weaker / partially contradicted claims

### Layer × parameter

| Layer / param | ShipsNet AAAD / ASD / ARIn | EuroSAT AAAD / ASD / ARIn |
|---|---|---|
| conv1 bias | .0093 / .0656 / **.0468** | .0281 / .1073 / .0785 |
| conv1 weight | .0124 / .0775 / .0555 | .0371 / .1181 / .0875 |
| conv2 bias | .0088 / .0791 / .0563 | .0298 / .1084 / .0795 |
| conv2 weight | .0130 / .1010 / .0720 | .0250 / .1013 / **.0738** |
| fc1 bias | .0450 / .1749 / .1277 | .0790 / .1797 / **.1388** |
| fc1 weight | .0571 / .1655 / .1238 | .0567 / .1478 / .1120 |

- "fc1 is the least robust layer" holds on both. The conv-vs-fc gap is smaller on
  EuroSAT (fc1 ARIn is ~1.8× conv, versus ~2.4× on ShipsNet) because conv-layer faults
  now also cost real accuracy.
- **"Weight is less robust than bias" (Section 4.6.2) is not clean.** It holds for
  conv1 and conv2 on both datasets, but reverses at fc1 on EuroSAT (bias `.1388` >
  weight `.1120`). On ShipsNet fold 1 the fc1 ordering is already mixed — weight has the
  worse AAAD (.0571 vs .0450) but the better ASD (.1655 vs .1749). The claim should be
  scoped to convolutional layers.

### Sign bit (index 0)

| Bit | ShipsNet ARIn | EuroSAT ARIn |
|---|---|---|
| 0 (sign) | .0394 | .0616 |
| 1 (exp MSB) | .3093 | .3331 |
| 3, 6 (exp) | .0386, .0387 | .0534, .0599 |
| 10, 15, 21 (mantissa) | .0385–.0388 | .0519–.0520 |

On ShipsNet, bit 0 is indistinguishable from the mantissa floor (.0394 vs ~.0386).
On EuroSAT it separates clearly (.0616 vs ~.0520), as does bit 6 (.0599). Sign and
low-exponent effects only become measurable in the multiclass setting — consistent with
the paper's observation (Section 4.7) that bit 0 yields a moderate log-absolute
difference but only a slight ASD increase.

### Original bit value

| Original bit | ShipsNet AAAD / ASD / ARIn (n) | EuroSAT AAAD / ASD / ARIn (n) |
|---|---|---|
| 0 | .0463 / .1801 / .1315 (1,466) | .0758 / .1732 / .1336 (1,722) |
| 1 | .0064 / .0542 / .0386 (1,810) | .0059 / .0761 / .0540 (1,554) |

The 1→0-is-safer conclusion replicates strongly. Note the sample split differs
(EuroSAT has more original-0 bits), reflecting a different learned weight magnitude
distribution — worth a sentence if this table goes in the paper.

## 5. Implications for the paper

**Safe to state generally** (replicate on both datasets): bit-position effects, original
bit value, layer depth / fc1 vulnerability, center-vs-spread, ReLU as least robust, RWG
as the AAAD leader, and the BNN confidence-stability mechanism.

**Must be scoped to ShipsNet** as currently written:

1. Abstract + Section 4.4.1 + Conclusion: "Gaussian prior is the most robust" — EuroSAT
   ranks Uniform first by ARIn.
2. Abstract + Section 4.5 + Conclusion: "Sigmoid was the most robust activation" —
   EuroSAT ranks it 5th of 7 with a severe accuracy penalty.
3. Section 4.6.2: "weight is less robust than bias" — holds for conv layers only.

**New material EuroSAT enables:**

- The multiclass amplification result itself: same architecture, same attack grid,
  AAAD +75% while ASD only +15%. This is a clean quantification of *where* BNN robustness
  comes from (confidence stability, not label stability).
- Prior scale `b` as a robustness axis (ARIn .0891 → .0993 across b = 0.1 → 10.0),
  which is the research angle flagged in `docs/analysis/shipsnet_vs_eurosat.md`.

## 6. Open items

- ShipsNet fold 1 has no `b ∈ {0.1, 10.0}` SEU runs, so the prior-scale trend is
  EuroSAT-only. A ShipsNet b-sweep at v00 would make it a two-dataset finding.
- EuroSAT variants 1–3 (smartpool / dropout / weight decay) are not covered here.
  The paper's strongest variant claim — dropout (Model 2) is the most robust BNN —
  is untested on EuroSAT. Note the SEU variant-switch bug tracked in memory
  (`task_shipsnet_seu_variant_switch`) affects the smartpool variant specifically.
- Superseded content: `docs/analysis/shipsnet_vs_eurosat.md` (2026-06-08) states
  "EuroSAT SEU not yet run" and reports accuracy-drop-only metrics from an earlier
  ShipsNet aggregation. This document supersedes its Section 2.
