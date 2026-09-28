---
type: results-report
date: 2026-06-14
experiment_line: eurosat-sweep-v02
round: 0
purpose: training-diagnostic (precheck b=10)
status: active
run_set_tag: eurosat-precheck-b10
model_variant: v02_00 (base — no smartpool, no dropout, no weight decay)
source_artifacts:
  - mlflow experiment: bnn-seu-eurosat (tag run_set=eurosat-precheck-b10)
linked_plan: docs/plan/eurosat-sweep-v02.md
---

# EuroSAT Sweep v02 / Round 0 / Training Diagnostic (Precheck b=10) / 2026-06-14

## 1. Executive Summary

The `eurosat-precheck-b10` run set (20 of 21 planned runs, b=10.0, model variant v02_00) reveals a **training failure** across most activation × prior configurations on the EuroSAT 10-class task. Virtually all Gaussian and Laplace configurations achieve near-chance accuracy (10–13%), while `tanh` is the only activation to show meaningful above-chance learning (24–27%). Uniform prior training is numerically unstable (ELBO overflows to `float_max`) in every case. One combination (`relu × laplace`) is missing entirely. These results indicate that a prior scale of b=10 is **too broad** for EuroSAT and must be investigated before the full sweep proceeds.

---

## 2. Experiment Identity

| Field | Value |
|-------|-------|
| Dataset | EuroSAT (10-class satellite imagery) |
| Model | BayesShipsCNN, `num_classes=10`, variant v02_00 (base) |
| Inference | Pyro SVI, Trace_ELBO, AutoNormal / AutoLaplace / AutoUniform |
| Optimizer | ClippedAdam, lr=1e-3 |
| Prior scale | b = 10.0 (μ = 0.0 fixed) |
| Priors tested | Gaussian, Laplace, Uniform |
| Activations tested | relu, tanh, sigmoid, sin, relu6, wg (_actWG), rwg (_actRWG) |
| Planned combos | 21 (7 activations × 3 priors) |
| Completed runs | 20 (relu × laplace missing) |
| MLflow tag | `run_set = eurosat-precheck-b10` |
| Run dates | 2026-06-10 to 2026-06-11 |

Chance baseline for 10-class EuroSAT: **10.0%**
Expected performance per sweep plan: **65–80%**

---

## 3. Results Table — Test Accuracy (%)

All results are for b = 10.0, variant v02_00 (base), 100 training epochs.

| Activation | Gaussian | Laplace | Uniform |
|:-----------|:--------:|:-------:|:-------:|
| relu       | 10.65    | —       | 19.96 † |
| tanh       | **26.76** | **24.22** | **25.78** † |
| sigmoid    | 10.59    | 10.13   | 24.00 † |
| sin        | 10.89    | 11.48   | 10.20 † |
| relu6      | 11.06    | 11.20   | 22.39 † |
| wg         | 12.00    | 10.91   | 17.57 † |
| rwg        | 11.98    | 10.98   | 15.43 † |

† Uniform prior: ELBO overflows to `float_max` in all cases (numerical instability). Accuracy values are reported but should not be interpreted as successful training.

— Missing: `relu × laplace` run not logged under `run_set=eurosat-precheck-b10`.

---

## 4. Training Accuracy (Final Epoch)

| Activation | Gaussian | Laplace | Uniform |
|:-----------|:--------:|:-------:|:-------:|
| relu       | 12.59    | —       | 15.80   |
| tanh       | 23.08    | 20.42   | 22.85   |
| sigmoid    | 13.03    | 12.24   | 24.82   |
| sin        | 13.00    | 12.56   | 12.57   |
| relu6      | 11.67    | 12.83   | 21.74   |
| wg         | 13.32    | 13.00   | 15.26   |
| rwg        | 13.02    | 13.08   | 14.88   |

The small gap between train and test accuracy across all configs confirms the models are not overfitting — they are simply not learning.

---

## 5. Key Observations

### 5.1 Near-Chance Performance (Gaussian & Laplace)

All Gaussian and Laplace configurations except `tanh` converge to 10–13% test accuracy — essentially the chance level for a 10-class balanced dataset. This is a training failure, not a regularisation outcome. The model is not extracting useful features from EuroSAT imagery at this prior scale.

**Likely cause:** b=10 is an extremely broad prior (±10 standard deviations from zero in Gaussian terms). For a 10-class task requiring more expressive feature maps than ShipsNet's binary task, this acts as excessive regularisation, keeping weights near zero and preventing the model from learning discriminative filters. The same b=10 on ShipsNet (binary) yielded 72–96% accuracy.

### 5.2 Uniform Prior — ELBO Overflow

Every uniform prior run produces an ELBO value of `1.797e+308` (Python's `float_max`), which is the result of a numerical overflow in the ELBO computation. This is a known issue with `AutoUniform` when the prior scale is very large. Test accuracy values for uniform runs are unreliable — the model checkpoint corresponds to a numerically degenerate training state.

**Action required:** Before including uniform prior in the full sweep, fix or gate the ELBO overflow. Options: (a) clip ELBO to a finite ceiling, (b) reduce b for uniform prior specifically, (c) exclude uniform from the b=10 tier.

### 5.3 Tanh is the Sole Surviving Activation

`tanh` achieves 24–27% test accuracy across all three priors, clearly above chance. This aligns with its bounded output range (−1 to +1), which may constrain activations into a range that remains meaningful under a very broad prior. All other activations either saturate (sigmoid, relu6) or produce unbounded/noisy activations (relu, sin, wg, rwg) that become uninformative under b=10.

### 5.4 Missing Run: relu × laplace

No MLflow run with `activation=relu, prior=laplace, prior_b=10.0` is logged under the `eurosat-precheck-b10` tag. Whether this run crashed, was skipped, or was logged to a different run_set is unknown. Must be re-run before the precheck is considered complete.

---

## 6. Comparison: EuroSAT b=10 vs ShipsNet b=10

| Config | ShipsNet test acc (b=10, 2025 backfill) | EuroSAT test acc (b=10, precheck) |
|:-------|:---------------------------------------:|:---------------------------------:|
| relu × gaussian | 86% | 10.7% |
| relu × uniform | 91% | 20.0% † |
| tanh × gaussian | 88% | 26.8% |
| tanh × uniform | 91–92% | 25.8% † |
| sigmoid × gaussian | 77–83% | 10.6% |
| relu6 × gaussian | 81–92% | 11.1% |

The contrast is stark. B=10 works reasonably well on ShipsNet (binary) but fails on EuroSAT (10-class). The task complexity increase is likely the dominant factor: a 10-class satellite imagery classifier requires more expressive weights, and b=10 prevents that.

---

## 7. Failure Cases and Limitations

| Issue | Scope | Severity |
|-------|-------|----------|
| Near-chance accuracy (Gaussian, Laplace) | 12 of 19 completed runs | Critical |
| ELBO overflow (Uniform) | 7 of 7 uniform runs | Critical |
| Missing relu × laplace run | 1 run | Minor |
| No multiple seeds — cannot distinguish training instability from systematic failure | All runs | Moderate |

---

## 8. What Changed Our Belief

Before this precheck: it was assumed that b=10 would provide a reasonable starting point for EuroSAT, mirroring the ShipsNet sweep structure.

After this precheck: b=10 is clearly insufficient as a prior scale for the EuroSAT 10-class task. The task complexity is high enough that the prior overwhelms the likelihood. The full b sweep (b ∈ {10, 1, 0.1}) is still planned, but the b=10 tier should be understood as a regularisation stress test rather than an expected good-performance regime. The b=1.0 and b=0.1 tiers are expected to be the operationally relevant ones.

---

## 9. Next Actions

| Priority | Action |
|:--------:|--------|
| 1 | **Re-run relu × laplace × b=10** to complete the 21-run precheck set |
| 2 | **Diagnose Uniform ELBO overflow** — check `AutoUniform` guide's ELBO computation at large b; consider capping or switching to a finite-range uniform |
| 3 | **Run b=1.0 tier** — this is the most likely operational prior scale for EuroSAT; run all 21 combos before committing to the full 63-run sweep |
| 4 | **Run b=0.1 tier** — tighter prior, may over-regularise but worth establishing the floor |
| 5 | **Run relu × laplace** across all b values in the full sweep to close the gap |
| 6 | **Proceed with full sweep only after b=1.0 shows plausible results** (target: ≥50% on at least 5 combos) |

---

## 10. SEU Evaluation Status

`scripts/eval_seu_eurosat.py` exists and handles Gaussian and Laplace priors.

**Current limitations of the SEU script relevant to this run set:**
- Uniform prior is **not supported** (`_build_guide` raises `ValueError` for `prior='uniform'`). This is consistent with the training instability observed above — uniform prior models from this precheck should not be SEU-evaluated until training is fixed.
- The script expects model artifacts under `results/eurosat/bayesian/` — confirm precheck artifacts were saved to `results/eurosat/bayesian/results_eurosat_v02_00/` before queuing SEU evaluation.
- With ~10% initial accuracy on most Gaussian/Laplace runs, SEU evaluation would not be meaningful — the model is already at floor performance.

**Recommendation:** Do not run SEU evaluation on precheck b=10 results. Wait for b=1.0 or b=0.1 runs with test accuracy ≥50% before investing SEU compute.
