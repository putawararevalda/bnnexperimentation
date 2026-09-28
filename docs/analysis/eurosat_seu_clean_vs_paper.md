# EuroSAT SEU (`seu_clean`, all 4 variants) vs. the ICAART paper's ShipsNet claims

> Generated: 2026-08-07
> Data: `results/eurosat/seu_clean/{v02_00,v02_01,v02_02,v02_03}` — 252 CSVs, 42,336 rows,
> **39,312 valid injections** (3,024 bit-0-on-scale rows are invalid and excluded).
> Paper: `docs/2026_ICAART_Revalda (14).pdf` (ShipsNet, BNN + DNN).
> Supersedes the variant-0-only comparison in `eurosat_v00_vs_shipsnet_fold1_seu.md`.

## Scope, alignment, and validity

**Grid.** EuroSAT `seu_clean` covers 4 model variants × 7 activations × 3 priors ×
3 prior scales `b ∈ {0.1, 1.0, 10.0}` = 252 model configs × 168 injections.
The paper's BNN grid is 4 variants × 7 activations × 3 priors at a **single** `b`.
`eurosat_v00_vs_shipsnet_fold1_seu.md` established that recomputing ShipsNet fold-1
variant-0 reproduces the paper's Table 3 BNN row exactly at `b = 1.0`, so **`b = 1.0`
(13,104 valid injections) is the primary, grid-matched slice** here. All-`b` pooled
numbers are reported as a sensitivity check; every verdict below is unchanged between
the two slices.

**Metrics follow the paper.** `AAAD = mean|acc_after − acc_before|`,
`ASD = mean softmax difference`, `ARIn = sqrt((AAAD² + ASD²)/2)` computed from the
**aggregated** AAAD and ASD.

**Variant switches are correctly applied.** `scripts/eval_seu_eurosat.py:387-390`
infers the variant from the search dir and rebuilds the model with
`smartpool_switch` / `dropout_switch` set accordingly. The ShipsNet variant-switch bug
tracked in memory does **not** affect this EuroSAT data — which is what makes the
variant claim testable for the first time.

### Blockers — what cannot be tested

1. **No EuroSAT deterministic-CNN SEU runs exist.** Every paper claim of the form
   "BNN vs DNN" (abstract headline `ARIn 0.0779 vs 0.1909`, Table 1, Table 2,
   Sections 4.2.x, and the DNN columns of Tables 3/5/6/7/8/9/10) is **untestable**
   on EuroSAT. Nothing below should be read as confirming or refuting them.
2. **n = 1 run per config, unseeded MC-10 inference.** There are no seeds or repeats,
   so run-to-run variance is unquantified. Paired tests below use the **model config**
   as the unit of analysis (which is legitimate — configs are independently trained),
   but any single aggregate difference smaller than ≈`.005` ARIn should be treated as
   within noise. This matters specifically for the variant comparison.
3. **`initial_accuracy` is the best-checkpoint MC-10 accuracy from the SEU CSVs** on
   both sides, so the last-epoch/best-checkpoint discrepancy noted in `CLAUDE.md` does
   not distort this comparison. It does mean these accuracies are not the ones in the
   training logs.

---

## 1. Headline numbers

| Slice | n valid | mean init acc | AAAD | ASD | ARIn |
|---|---|---|---|---|---|
| **Paper, ShipsNet BNN** (Table 2) | 13,104 | .8285 | .0244 | .1074 | **.0779** |
| **EuroSAT, `b=1.0`** | 13,104 | .7935 | .0426 | .1265 | **.0944** |
| EuroSAT, all `b` pooled | 39,312 | .8018 | .0417 | .1269 | .0944 |

EuroSAT BNNs are **~21% less robust by ARIn**, and the gap is asymmetric:
**AAAD +75%** (.0244 → .0426) but **ASD only +18%** (.1074 → .1265). This is the
expected 10-class amplification — a perturbed logit vector has nine ways to cross a
decision boundary instead of one — and it is the single most important framing fact
for reading everything below. The paper's *mechanism* for BNN robustness (MC averaging
stabilises predicted probabilities, Section 4.2.3) transfers; its *accuracy-side*
consequences weaken.

---

## 2. Verdict table

| # | Paper claim (ShipsNet) | EuroSAT verdict |
|---|---|---|
| 1 | BNN more robust than DNN (ARIn .0779 vs .1909) | **untestable** — no EuroSAT DNN runs |
| 2 | Dropout (Model 2) is the most robust BNN variant; "largest robustness gain" | **does not hold** — direction survives, effect vanishes (p = .76) |
| 3 | Gaussian prior is the most robust | **does not hold — inverts.** Uniform wins (p = 1.5e-6) |
| 3a | Uniform has the worst AAAD | **holds** (.0447 vs .0416 / .0414) |
| 3b | Uniform has the best ASD under spread attack (§4.4.3) | **holds, and strengthens** — best ASD on *both* C and S |
| 4 | Prior **center** is more robust than prior **spread** | **holds** — every prior, both metrics |
| 5 | Sigmoid is the most robust activation (best ASD + ARIn) | **does not hold** — 5th/7 ARIn, 6th/7 ASD |
| 6 | RWG has the best AAAD | **partially** — 2nd; WG takes 1st |
| 7 | ReLU is the least robust activation | **holds** — worst on all three metrics |
| 8 | ReLU6 meaningfully better than ReLU | **weakens badly** — .1194 vs .1298 (was .0737 vs .1180) |
| 9 | Accuracy ↔ robustness trade-off | **contradicted — sign flips.** ρ = −0.86 across activations |
| 10 | fc1 least robust, conv2 most robust | **holds** |
| 11 | Weight is less robust than bias (§4.6.2) | **does not hold** — reverses at conv2 *and* fc1 |
| 12 | Bit 1 (exponent MSB) is the most devastating | **holds, strongly** |
| 13 | Mantissa flips are negligible | **holds** |
| 14 | Bit 0 (sign) is only slightly above the mantissa floor | **holds directionally, effect is larger** |
| 15 | 1→0 flips far more robust than 0→1 | **holds, strongly** |
| — | Prior scale `b` degrades robustness monotonically | *EuroSAT-only, no ShipsNet counterpart* |

---

## 3. Claims that replicate

### 3.1 Bit position (Tables 8/9) — replicates cleanly

| Bit | Paper BNN AAAD / ASD / ARIn | EuroSAT AAAD / ASD / ARIn |
|---|---|---|
| 0 (sign) | .0075 / .0533 / .0381 | .0127 / .0860 / .0615 |
| **1 (exp MSB)** | .1206 / .4080 / **.3009** | .2381 / .4003 / **.3293** |
| 3 (exp) | .0070 / .0527 / .0376 | .0071 / .0765 / .0544 |
| 6 (exp) | .0070 / .0529 / .0378 | .0127 / .0863 / .0617 |
| 10 / 15 / 21 (mantissa) | .0067 / .0525 / .0374–.0376 | .0041–.0042 / .0720–.0721 / **.0510–.0511** |

Bit 1 is catastrophic on both (**6.4× the floor** on EuroSAT vs 8.0× on ShipsNet);
mantissa bits sit at an indistinguishable floor on both. Two refinements: EuroSAT's
AAAD at bit 1 is **double** ShipsNet's (.2381 vs .1206) while ASD is identical
(.4003 vs .4080) — again the multiclass accuracy amplification. And bit 0 / bit 6
separate from the mantissa floor on EuroSAT (.0615 / .0617 vs .0511) where on ShipsNet
they did not (.0381 / .0378 vs .0374). The paper's Section 4.7 remark that bit 0 gives a
moderate log-absolute difference but only a slight ASD bump is **more** visible here,
not less.

### 3.2 Original bit value (Table 10) — replicates strongly

| Original bit | Paper BNN ARIn | EuroSAT ARIn (n) |
|---|---|---|
| 0 (0→1 flip) | .1268 | **.1338** (6,860) |
| 1 (1→0 flip) | .0378 | **.0532** (6,244) |

AAAD tells the same story: .0455 → .0069 (paper) vs .0759 → .0060 (EuroSAT). The paper's
disagreement with (Hanif et al., 2021) is reproduced on a second dataset — worth stating,
because a two-dataset contradiction of published work is much harder to dismiss as a
ShipsNet artifact.

### 3.3 Layer depth (Tables 6/7) — replicates

| Layer / param | Paper BNN ARIn | EuroSAT ARIn |
|---|---|---|
| conv1 bias | .0458 | .0810 |
| conv1 weight | .0546 | .0887 |
| conv2 bias | .0515 | .0799 |
| **conv2 weight** | .0704 | **.0719** (most robust) |
| fc1 weight | .1205 | .1056 |
| **fc1 bias** | .1259 | **.1404** (least robust) |

conv2 most robust, fc1 least robust — holds on both. The gap narrows (fc1 ≈ 1.7× conv on
EuroSAT vs ≈ 2.4× on ShipsNet) because conv-layer faults now cost real accuracy in the
multiclass setting, raising the conv floor.

### 3.4 Prior center vs. prior spread (Table 4, §4.4.4) — replicates for every prior

| Prior | AAAD C / S | ASD C / S | ARIn C / S |
|---|---|---|---|
| Gaussian | .0375 / .0464 | .1279 / .1356 | .0943 / **.1014** |
| Laplace | .0377 / .0457 | .1367 / .1455 | .1003 / **.1079** |
| Uniform | .0331 / .0583 | .0972 / .1191 | .0726 / **.0938** |
| **pooled** | **.0361 / .0502** | **.1206 / .1334** | **.0890 / .1008** |

"For all metrics, the center prior parameter is consistently more robust than the spread
prior parameter" holds without exception. The mechanism (a mean shift moves all MC
samples by the same amount and cancels under averaging; a scale shift does not) is
dataset-independent, as expected.

### 3.5 ReLU is the least robust activation — replicates

EuroSAT: relu is worst on AAAD (.0736), ASD (.1682) and ARIn (.1298), 12/12 configs.
Same on ShipsNet. Safe to state generally.

---

## 4. Claims that do NOT hold

### 4.1 "Dropout yielded the largest robustness gain" — effect vanishes

| Variant | Paper BNN AAAD / ASD / ARIn | EuroSAT AAAD / ASD / ARIn |
|---|---|---|
| 0 base | .0243 / .1106 / .0800 | .0426 / .1271 / .0948 |
| 1 smartpool | .0254 / .1109 / .0805 | .0427 / .1281 / .0955 |
| **2 dropout** | .0221 / .0944 / **.0685** | .0425 / .1242 / **.0929** |
| 3 weight decay | .0257 / .1138 / .0825 | .0425 / .1266 / .0944 |

Dropout is still nominally best on EuroSAT, but the gain collapses from
**−14.4%** vs base (−.0115 ARIn) to **−2.0%** (−.0019). AAAD is flat to four decimals
across all four variants (.0425–.0427) — on ShipsNet dropout's AAAD advantage was
9% below base.

Paired test, unit = model config, pairing on (activation × prior), n = 21:

| Contrast | mean ΔARIn | configs improved | Wilcoxon p |
|---|---|---|---|
| dropout − base | −.0022 | 12/21 | **.76** |
| weight_decay − base | −.0004 | 9/21 | 1.00 |
| smartpool − base | +.0006 | 10/21 | .86 |

None is significant, and the per-config winner is split almost evenly
(base 6, dropout 5, smartpool 5, weight_decay 5 of 21). Given blocker #2 (n=1 run,
unseeded MC), a −.0019 aggregate difference is **not distinguishable from run noise**.

**Verdict:** the *direction* survives, the *claim* does not. On EuroSAT, architecture
variant is not a meaningful robustness lever — the mechanism argued in Section 4.3.3
(dropout removes dominant neurons) does not produce a measurable effect with 10 classes.
This is the first test of the paper's strongest variant claim on a second dataset, and
it is the one I would flag hardest before submission.

### 4.2 "Gaussian priors outperforming Laplace and Uniform" — inverts

| Prior | Paper BNN ARIn (C / S) | EuroSAT AAAD / ASD / **ARIn** |
|---|---|---|
| Gaussian | .0575 / .0929 | .0416 / .1315 / .0975 |
| Laplace | .0598 / .0946 | .0414 / .1408 / .1038 |
| **Uniform** | .0763 / .0928 | .0447 / .1073 / **.0822** |

Uniform is the most robust prior on EuroSAT, and it is not marginal: paired on
(activation × variant), n = 28, uniform beats Gaussian in **24/28** configs
(mean ΔARIn −.0153, **p = 1.5e-6**) and beats Laplace in **28/28**
(−.0215, p = 7.5e-9). Laplace is worst, where the paper had it second.

The driver is entirely ASD. The paper's own Section 4.4.3 mechanism predicts this — a
flat PDF means a scale perturbation changes sampling probability less than for a peaked
Gaussian/Laplace — it simply was not strong enough to dominate ARIn in a binary task.
Two sub-claims **do** survive: Uniform still has the worst AAAD (.0447), and Uniform
still has the best spread-attack ASD (.1191 vs .1356 / .1455). What breaks is only the
ARIn-level ranking, i.e. the sentence in the abstract and conclusion.

### 4.3 "Sigmoid was the most robust activation" — drops to 5th and pays a severe accuracy cost

| Activation | Paper BNN AAAD / ASD / ARIn | EuroSAT AAAD / ASD / ARIn | EuroSAT init acc |
|---|---|---|---|
| **WG** | .0213 / .0964 / .0698 | **.0240** / **.0910** / **.0665** | **.8482** |
| RWG | **.0168** / .0958 / .0688 | .0286 / .0963 / .0710 | .8217 |
| sin | .0294 / .1056 / .0775 | .0290 / .1026 / .0754 | .8250 |
| tanh | .0214 / .1102 / .0794 | .0309 / .1147 / .0840 | .8181 |
| **sigmoid** | .0190 / **.0800** / **.0581** | .0455 / .1575 / .1159 | **.6777** |
| relu6 | .0248 / .1012 / .0737 | .0664 / .1553 / .1194 | .7825 |
| relu | .0379 / .1626 / .1180 | .0736 / .1682 / .1298 | .7811 |

Sigmoid falls from **1st of 7** on ShipsNet to **5th of 7** on EuroSAT, and its baseline
accuracy collapses to .678 — 10–17 points below every other activation. The paper's
mechanism (Section 4.5: `h′ ≈ 0.25` near zero damps weight perturbations) is real, but
the same saturation that damps perturbations also damps learning; with 10 classes the
cost stops being affordable. Related: **ReLU6's advantage over ReLU largely disappears**
(.1194 vs .1298 on EuroSAT; .0737 vs .1180 on ShipsNet) — clipping at 6 stops helping once
the pre-activation range needed for 10-class separation is wider.

**RWG best AAAD** is *partially* true: on EuroSAT WG takes both AAAD (.0240) and ARIn
(.0665), with RWG 2nd on both. The generalisable version of the claim is
"the Dennis & Pope activations (WG/RWG) lead on AAAD", not "RWG is best".

### 4.4 The accuracy–robustness trade-off — sign flips

The paper argues (Section 4.5, echoed in the conclusion and abstract) that robustness is
bought with accuracy: ReLU had the best accuracy and worst robustness, RWG the worst
accuracy and best robustness.

On EuroSAT this reverses. Across 84 model configs, initial accuracy and ARIn are
**negatively** correlated — more accurate models are *more* robust:

- per-config: Spearman ρ(init acc, ARIn) = **−0.655**, p = 1.4e-11 (n = 84)
- per-config: Spearman ρ(init acc, AAAD) = **−0.639**, p = 6.3e-11
- per-activation: ρ = **−0.857**, p = .014 (n = 7)

WG is simultaneously the **most accurate** (.8482) and **most robust** (.0665)
activation; sigmoid is the least accurate (.6777) and 5th most robust. The ShipsNet
trade-off was an artifact of sigmoid happening to sit at a good accuracy/robustness
point in a binary task. On EuroSAT there is no trade-off to report — there is an
alignment.

**This is the claim most at risk of a reviewer challenge**, because the paper cites
(Su et al., 2018) and (Carbone et al., 2020) in support of it, and a second dataset
from the same codebase produces the opposite sign at p < 1e-10.

### 4.5 "Weight is less robust than bias" — reverses at 2 of 3 layers

| Layer | ARIn weight | ARIn bias | configs where weight is worse | Wilcoxon p |
|---|---|---|---|---|
| conv1 | .0900 | .0820 | 46/84 | .0069 ✔ supports claim |
| conv2 | .0728 | .0808 | 16/84 | 4.3e-08 ✘ reverses |
| fc1 | .1061 | .1408 | 14/84 | 2.2e-11 ✘ reverses |
| **pooled** | **.0886** | **.1002** | — | ✘ reverses |

Pooled across the network the claim is **backwards** on EuroSAT: bias attacks are worse
(.1002 vs .0886). Only conv1 supports the paper's amplification argument (Section 4.6.2:
`w` is multiplied by `x_i`, `b` is a constant), and even there the config-level win rate
is 46/84 — the aggregate is driven by magnitude, not consistency.

Note this is *stricter* than the variant-0-only finding in
`eurosat_v00_vs_shipsnet_fold1_seu.md`, which reported the reversal at fc1 only. On the
full clean grid conv2 reverses too. The claim should be dropped or scoped to conv1.

---

## 5. EuroSAT-only material (no ShipsNet counterpart)

**Prior scale `b` is a robustness axis**, and it trades directly against accuracy:

| `b` | mean init acc | AAAD | ASD | ARIn |
|---|---|---|---|---|
| 0.1 | .6734 | .0369 | .1198 | **.0886** |
| 1.0 | .7935 | .0426 | .1265 | .0944 |
| 10.0 | .8023 | .0455 | .1343 | .1003 |

Robustness degrades monotonically as the prior widens (+13% ARIn from `b`=0.1 to 10),
while accuracy improves monotonically. Note `b`=0.1 includes configs that fail to train
(min init acc .1648 ≈ 1/10 chance), so part of that "robustness" is a degenerate model
that cannot be perturbed away from its already-poor predictions — **this table should not
be presented as a clean robustness knob without filtering non-converged configs.**

---

## 6. What this means for the paper

**Safe to state as general (two-dataset) findings:**
bit-position effects (bit 1 catastrophic, mantissa negligible), original-bit-value
asymmetry incl. the contradiction of (Hanif et al., 2021), layer depth / fc1
vulnerability, prior center more robust than prior spread, ReLU as least robust, and
WG/RWG as the AAAD leaders.

**Must be scoped to ShipsNet as currently written:**

1. Abstract + §4.4.1 + Conclusion — "Gaussian prior is the most robust". EuroSAT ranks
   Uniform first by a wide, highly significant margin.
2. Abstract + §4.5 + Conclusion — "Sigmoid was the most robust activation despite an
   accuracy trade-off". EuroSAT: 5th of 7 with a 10–17 point accuracy penalty.
3. Abstract + §4.3.2 + Conclusion — "Dropout yielded the largest robustness gain".
   EuroSAT: no significant variant effect at all (p = .76).
4. §4.6.2 — "weight is less robust than bias". Reverses at conv2 and fc1; pooled it is
   backwards.
5. §4.5 + Conclusion — the accuracy/robustness trade-off. EuroSAT shows the opposite
   sign at ρ = −0.66, p = 1.4e-11.

**Untestable until DNN SEU runs exist on EuroSAT:** the entire BNN-vs-DNN comparison,
which is the paper's headline result.

**New material EuroSAT enables:** the multiclass amplification quantification (identical
architecture and attack grid; AAAD +75% while ASD only +18%), which is a clean
demonstration that BNN SEU robustness lives in *confidence stability*, not *label
stability*.

## 7. Open items

1. **Run the deterministic-CNN SEU grid on EuroSAT.** Without it the headline claim
   cannot be generalised, and a reviewer asking "does this hold beyond binary
   classification?" cannot be answered on the paper's main result.
2. **Add seeds/repeats.** With n = 1 and unseeded MC-10, the variant comparison
   (differences ≈ .002 ARIn) is under-powered by construction. At minimum, 3 seeds on
   the 4 variants at one (activation, prior) would bound the noise floor.
3. **ShipsNet `b`-sweep** to make the prior-scale trend a two-dataset finding, filtering
   non-converged configs.
4. The ShipsNet SEU variant-switch bug (memory: `task_shipsnet_seu_variant_switch`) means
   the paper's *ShipsNet* smartpool row may itself be wrong; the EuroSAT side is clean.
   Re-check before using Table 3 row 1.

**Reproduce:**
`uv run --with pandas,numpy,scipy python scripts/compare_seu_datasets.py` (variant-0 doc),
or the ad-hoc aggregation described in §Scope above over `results/eurosat/seu_clean/`.
