# Evolution of Claims: Rejected Draft vs. Corrected Analysis

The rigorous methodological corrections (5-fold cross-validation, proper model checkpoint loading, and fixing the double-softmax bug) have fundamentally altered the narrative of the paper. Below is a detailed comparison of the claims made in the original rejected submission versus what the data actually proves today.

## 1. Headline Claim: BNN vs. Deterministic (DNN) Robustness
* **Rejected Paper:** Claimed that BNNs were significantly more robust against SEUs than deterministic models (reporting ARIn 0.0779 for BNN vs. 0.1909 for DNN). It also claimed BNNs suffered a massive ~15% accuracy penalty (82.8% vs 97.5%) to achieve this.
* **Latest Comparison:** **Completely Overturned.** With the DNN's double-softmax bug fixed, the DNN achieves an incredibly robust ARIn of **0.0156**, while the absolute best BNN manages only **0.0572**. Furthermore, the BNN accuracy penalty is only ~3-7% (90-94% vs 97.5%). 
* **New Narrative:** While BNNs are known to be robust against *external* input noise (e.g., adversarial attacks), they are mathematically *more fragile* to *internal* parameter corruption (SEUs) than standard deterministic models.

## 2. Model Variants (Ablation Study)
* **Rejected Paper:** Claimed that adding Dropout yielded the largest robustness gain among the Bayesian variants.
* **Latest Comparison:** **Overturned (Null Result).** Dropout, smart pooling, and weight decay provide zero meaningful robustness benefit. Across both datasets, the ARIn spread between all variants is less than 5%. 
* **New Narrative:** Structural regularizers intended to prevent overfitting do not inherently protect against bit-level parameter corruption.

## 3. Activation Functions
* **Rejected Paper:** Concluded that Sigmoid and Rectified Weighted Gaussian (RWG) were the most robust activations.
* **Latest Comparison:** **Overturned.** Sigmoid actually performs terribly on accuracy. Instead, **Sinusoidal** strictly dominates on ShipsNet (best ARIn and Accuracy), while **Weighted Gaussian (WG)** dominates on EuroSAT. 
* **New Narrative:** Periodic and bounded activations consistently outclass unbounded functions like ReLU, but the exact optimal activation is dataset-dependent.

## 4. Prior Distributions
* **Rejected Paper:** Stated that the Gaussian prior was strictly the most robust.
* **Latest Comparison:** **Nuanced/Corrected.** The Uniform prior is actually the most robust (lowest ARIn), but it causes a severe drop in accuracy. 
* **New Narrative:** Gaussian isn't the most robust in an absolute sense, but it offers the only viable *balance* of high accuracy and reasonable robustness.

## 5. Injection Site (Layer & Bit Position)
* **Rejected Paper:** Vaguely noted that fully connected layers were worse, and that bit index 1 caused severe issues.
* **Latest Comparison:** **Validated & Amplified.** We confirmed `fc1` is universally the most vulnerable layer. More importantly, we sharpened the mechanism: flipping **Bit 1 (Exponent MSB)** causes a catastrophic failure roughly 10x worse than any mantissa bit because it rescales the IEEE-754 float by a massive factor of $2^{128}$.
