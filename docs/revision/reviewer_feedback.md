# ICAART 2026 — Paper #90 Revision Checklist

## Reviewer #1 (Overall: 2/6)

### Major Issues

- [ ] **Justify ARIn** — Explain *why* the metric is needed and *why* it is computed as the RMS of AAD and Softmax Difference. Currently the paper only states "to make comparison easier" with no mathematical or conceptual motivation.
- [ ] **Add more datasets** — ShipsNet alone makes findings dataset-specific. Add at least one more dataset (EuroSAT is already available in the codebase).
- [ ] **Multiple training runs** — A single 80/20 split is insufficient. Use k-fold cross-validation (k=10 suggested) to show results are not artifacts of a single random seed or optimization path.
- [ ] **Address the DNN vs BNN AAD contradiction** — Table 2 shows DNNs are more robust on AAD. Since AAD has smaller magnitude than Softmax Difference, it barely affects ARIn, making the "BNNs are more robust overall" conclusion debatable. Discuss this explicitly.

### Minor Issues

- [ ] **Typo** — p.1 col.1 last line: "reducing bandwid" → "reducing bandwidth"
- [ ] **Typo** — p.4 col.1: "The model will is trained on the dataset" → fix grammar
- [ ] **Layout** — Equation (5) goes out of page bounds; fix formatting
- [ ] **Missing citations** — Figures 3 and 4 are not cited anywhere in the text
- [ ] **Missing citation** — Table 1 is not cited in the text; add a reference and clarify that it shows mean results across all activation/model variant combinations
- [ ] **Add std to Table 1** — Include standard deviation to illustrate result variability across model variants
- [ ] **Layout** — Tables 3, 4, and 5 go out of page bounds; fix formatting
- [ ] **SEU count discrepancy** — p.5 col.2: the same paragraph mentions 15,456 and 13,104 SEU injections — clarify and correct

---

## Reviewer #2 (Overall: 2/6)

### Major Issues

- [ ] **Articulate novelty clearly** — The paper does not explicitly state what is new. Add a clear novelty statement (e.g., first study of BNN robustness under weight/bias SEU perturbation, novel ARIn metric).
- [ ] **Strengthen background / state of the art** — The background section does not adequately cover the current literature or formally define the problem. Add recent related work and a problem formalization.
- [ ] **Add ablation study** — Include analysis of individual design choices (e.g., effect of dropout alone, effect of prior alone) to isolate contributions.
- [ ] **Add hyperparameter analysis** — Discuss the sensitivity of results to key hyperparameters (learning rate, number of MC samples, prior scale `b`).
- [ ] **Add SOTA comparison** — Compare against at least one existing method for neural network robustness under weight perturbation.
- [ ] **Shorten abstract and introduction** — Both are flagged as too long.
- [ ] **Improve figures** — Reviewer flagged figures as inadequate in number or quality; review and improve.
- [ ] **Update references** — Add more recent (2023–2025) references relevant to BNN robustness and SEU mitigation.
- [ ] **Improve English throughout** — Full language/grammar pass needed.
