# ShipsNet vs EuroSAT — Results Comparison & Analysis

> Generated: 2026-06-08 | Based on results in `results/shipsnet/` and `results/eurosat/`

---

## 1. Baseline Accuracy

### ShipsNet

| Model Type | Best Accuracy | Config |
|---|---|---|
| Deterministic CNN | **~99.4%** | relu, val accuracy epoch 4+ |
| BNN — best | **93.4%** | actRWG / uniform / b=0.1 |
| BNN — median | ~78–87% | relu or sin / uniform / b=0.1–1.0 |
| BNN — worst | ~60.4% | actWG / laplace / b=0.1 |

**Top ShipsNet BNN configs (by best_accuracy, results_shipsnet_v02_01):**

| Activation | Prior | b | Accuracy |
|---|---|---|---|
| actRWG | uniform | 0.1 | 93.4% |
| sin | uniform | 0.1 | 93.1% |
| actWG | uniform | 0.1 | 93.0% |
| relu | uniform | 0.1 | 92.9% |
| relu6 | uniform | 0.1 | 92.8% |
| tanh | uniform | 0.1 | 91.3% |
| relu | gaussian | 1.0 | 86.6% |

### EuroSAT

| Config | b | Accuracy | Notes |
|---|---|---|---|
| relu / gaussian | **10.0** | **81.2%** | 100 epochs, converges steadily |
| relu / laplace | 10.0 | 78.3% | 100 epochs |
| tanh / gaussian | 10.0 | 54.7% | 10 epochs only |
| relu6 / gaussian | 10.0 | 52.3% | 10 epochs only |
| tanh / gaussian | 1.0 | ~29% | Plateaus immediately |
| relu / uniform | 1.0 | ~24% | Plateaus immediately |
| relu / laplace | 1.0 | ~13% | Random-guess level (~1/10 chance) |
| Any config | 0.1 | ~13% | Completely stuck at random |

---

## 2. SEU Robustness (ShipsNet only — EuroSAT SEU not yet run)

### By Prior

| Prior | Mean Accuracy Drop | Verdict |
|---|---|---|
| Uniform | **−0.06%** | Most robust |
| Laplace | −1.35% | Moderate |
| Gaussian | −1.49% | Least robust |

Uniform prior's bounded support constrains weight magnitude, making individual bit corruptions less impactful.

### By Layer

| Layer | Mean Accuracy Drop | Mean Softmax Diff |
|---|---|---|
| conv1 | 0.08% | 0.256 |
| conv2 | 0.10% | 0.258 |
| **fc1** | **2.77%** | **0.276** |

The fully connected layer is ~34× more sensitive than convolutional layers. A single-bit flip in fc1 weights can directly change the class logit.

### By Bit Position

| Bit | Mean Accuracy Drop | Notes |
|---|---|---|
| 0 | −0.25% | Sign bit — moderate |
| **1** | **+8.83%** | **Catastrophic — exponent MSB** |
| 3 | −0.23% | Negligible |
| 6 | −0.24% | Negligible |
| 10 | −0.25% | Negligible |
| 15 | −0.22% | Negligible |
| 21 | −0.23% | Negligible |

Bit 1 (MSB of the FP32 exponent) causes catastrophic degradation because flipping it multiplies or divides the weight value by ~2^128, producing ±∞ or subnormal values that collapse the entire output distribution.

---

## 3. Why Is EuroSAT Accuracy Low?

### Short answer: it is not inherently low — the prior scale `b` is the culprit.

The single run with `relu / gaussian / b=10.0` over 100 epochs reached **81.2%**, which is a reasonable baseline for a BNN on a 10-class satellite image task. The majority of EuroSAT experiments used `b=1.0` or `b=0.1`, which caused them all to fail.

### Root Causes

#### 1. Prior scale `b` is too tight (primary cause)
- With `b=0.1` → all configs stuck at ~13% (random chance for 10 classes)
- With `b=1.0` → most configs plateau at 13–29%
- With `b=10.0` → relu/gaussian reaches 81.2%

A tight prior (small `b`) acts as very strong regularization. The posterior is pulled so hard toward zero that the model cannot learn discriminative features. For a 10-class task with subtle inter-class differences (e.g., highway vs. river from above), the weights need more expressive range.

ShipsNet (binary) gets away with `b=0.1` because even weakly-learned features are enough to separate ship/no-ship. EuroSAT needs the network to simultaneously distinguish 10 classes — the weights need larger magnitude to do that.

#### 2. Architecture may be underpowered for EuroSAT
The model (`conv1(3→32,k=3) → conv2(32→64,k=3) → fc1(16384→10)`) was designed for binary ShipsNet. EuroSAT has 10 finer-grained classes. A deeper or wider network would likely help, but this is a secondary concern — the prior scale issue is more fundamental.

#### 3. Batch size mattered in older experiments
The successful `b=10` run used `batch_size=54`, which gives more stable ELBO gradient estimates per step. The "newslate" runs used `batch_size=16`, adding noise to already-struggling training.

#### 4. Not enough epochs for hard configs
Most EuroSAT experiments only ran 10 epochs (aside from the `b=10` relu runs which ran 100). The accuracy curves show EuroSAT needs significantly more epochs to converge. The relu/gaussian/b=10 curve went from 37% at epoch 1 → 81% at epoch 100 — still climbing.

---

## 4. Inspiration: What to Try Next

### Immediate (fix the EuroSAT results)
- **Re-run EuroSAT with `b=10.0` across all activations** — this is the only confirmed working scale. Current results only cover relu and tanh with b=10.
- **Run EuroSAT SEU evaluation** — `scripts/eval_seu_eurosat.py` is ready; you just need trained models at b=10.
- **More epochs** — run 200+ epochs for EuroSAT; the accuracy curve at b=10 was still rising at epoch 100.

### Research angle: prior scale as a robustness variable
The ShipsNet results show uniform prior (bounded) is most SEU-robust. For EuroSAT, the question becomes: does a wider prior (b=10, needed for accuracy) also change the SEU robustness profile? This is a direct paper contribution — **the accuracy-robustness trade-off as a function of prior scale**.

### Architecture improvement
A simple upgrade: add a third conv layer (`conv3: 64→128`) before fc1. This halves fc1's input dimension (from 16384→8192 with one more pool) and adds more representational capacity for 10-class discrimination. Since fc1 is the most SEU-vulnerable layer, reducing its size is also a robustness benefit.

### SmartPool for EuroSAT
SmartPool was designed for ShipsNet SEU resilience. If you run EuroSAT SEU experiments, testing SmartPool (`smartpool_switch=True`) on EuroSAT could be a direct comparison point in the paper.
