# Robustness Metric Citation Audit

**Date:** 2026-08-11
**Scope:** Every robustness metric used in the ICAART paper (`docs/2026_ICAART_Revalda (14).pdf`)
and the MSc dissertation (`docs/dissertation/PutawaraRevalda_mscthesis (24).pdf`), checked
against the work it cites.
**Method:** LaTeX source extracted from `docs/dissertation/`; each cited work verified against
its canonical record (NeurIPS/ECCV/IJCAI proceedings, publisher DOI, arXiv landing page).
Claims were checked against the source text, not just metadata.

## Summary

| Metric | Reference status |
|---|---|
| Accuracy Delta → AAD | Concept is standard; the citation set is thin. Absolute-value modification is the author's own and is correctly flagged as such. |
| Softmax Difference | Solid — formula verified verbatim in the cited source. Upstream reference missing; equation is written for the wrong perturbation type. |
| ARIn | No reference, and none claimed. Reviewer #1 has asked for a justification. |

---

## 1. Softmax Difference — verified, exact match

Source: Carbone, Wicker, Laurenti, Patane, Bortolussi, Sanguinetti (2020),
*Robustness of Bayesian Neural Networks to Gradient-Based Attacks*.

§5.3 of the NeurIPS PDF states verbatim:

> "for a collection of N test point, we compute
> $\frac{1}{N}\sum_{j=1}^{N}\big|\langle f(x_j,w)\rangle_{p(w|D)} - \langle f(\tilde{x}_j,w)\rangle_{p(w|D)}\big|_\infty$"

This matches ICAART Eq. (9) and thesis Eq. `eq:softmaxdifference` exactly.

Two supporting claims also verified in the same paragraph:

- "robustness (i.e., 1 - softmax difference)" appears literally — the thesis's phrasing is accurate.
- Carbone cite Su et al. 2018 for the accuracy/robustness trade-off, so the
  `isrobustnessdongsu2018` attribution is correct and placed correctly.

### Issue 1.1 — upstream reference missing

Carbone do not claim the metric as their own. They justify it by reference to
**Cardelli et al., IJCAI 2019, "Statistical Guarantees for the Robustness of Bayesian
Neural Networks"** — "as this provides a quantitative and smooth measure of adversarial
robustness that is closely related with mis-classification ratios [Cardelli et al., 2019a]".

Citing Cardelli alongside Carbone materially strengthens the metric's provenance. As it
stands, Softmax Difference reads as a one-paper metric.

### Issue 1.2 — the equation describes the wrong perturbation (correctness bug)

Carbone perturb the **input**: $f(x_j,w)$ vs $f(\tilde{x}_j,w)$, under the *same* posterior
$p(w|D)$. This work perturbs the **weights**. Both ICAART Eq. (9) and thesis
Eq. `eq:softmaxdifference` still carry $\tilde{x}_j$, copied verbatim from the
input-attack setting.

The equation as printed does not describe the experiment that was run. It should compare the
same input under the original and SEU-perturbed guide, e.g.

$$\text{Softmax Difference} = \frac{1}{N}\sum_{j=1}^{N}\Big|\langle f(x_j,w)\rangle_{q_\phi(w)} - \langle f(x_j,w)\rangle_{q'_\phi(w)}\Big|_\infty$$

This is worth fixing ahead of anything else in this document. A careful reviewer will catch
it, and it undermines the paper's central novelty claim (weight/bias perturbation rather than
input perturbation) at precisely the point where that distinction should be emphasised.

---

## 2. AAD — concept is standard, citations are the weak part

Cited for accuracy delta: Dennis & Pope 2025, Feng et al. 2024, Pang et al. 2021. All three
exist and all three do measure accuracy under perturbation, so the claim is true. Their
individual strength varies:

| Citation | Status |
|---|---|
| Dennis & Pope, ICAART 2025 | Verified, DOI `10.5220/0013155000003890`. Direct predecessor work. Strongest of the three. |
| Feng et al. 2024, *Attacking Bayes* | Verified. Currently cited as a bare arXiv `@misc` — it was **published in TMLR (2024)** and received a TMLR Certification. Cite the TMLR version. |
| Pang et al. 2021 | Verified as arXiv:2106.09223, but **preprint only, never peer-reviewed**. Weakest link. |

For a claim as strong as "the most commonly used measure of robustness in this field," the
support is one conference paper and two preprints. Add a canonical anchor — Su et al.
ECCV 2018 is already in the bib, and Yan et al. ASP-DAC 2020 fits the weight-perturbation
side specifically, since SIPP is accuracy delta under bitflips.

The absolute-value modification (penalise positive and negative accuracy change equally,
without cancellation) is the author's own and is correctly presented as a modification. No
citation needed.

---

## 3. ARIn — genuinely unreferenced

No prior work is cited, and no equivalent metric was found in the literature. That is
acceptable as a novelty claim, but Reviewer #1's request is currently unanswered:

> "Justify ARIn — Explain *why* the metric is needed and *why* it is computed as the RMS of
> AAD and Softmax Difference."

### Suggested grounding: multi-criteria decision analysis

ARIn is the (scaled) Euclidean distance from the ideal point $(0,0)$ in the
(AAD, Softmax Difference) plane. That is exactly the separation measure used in **TOPSIS**
— Hwang, C.-L. & Yoon, K. (1981), *Multiple Attribute Decision Making: Methods and
Applications*, Springer, Lecture Notes in Economics and Mathematical Systems vol. 186.

This gives a well-established citation for the aggregation form and answers "why RMS rather
than a mean": RMS penalises a large failure on one criterion instead of letting a good score
on the other mask it.

### But TOPSIS also exposes the scale problem

TOPSIS **normalises criteria before combining them**. ARIn does not. Observed magnitudes:

| Quantity | Range in the paper |
|---|---|
| AAAD | 0.0149 – 0.0244 |
| ASD | 0.1074 – 0.2696 |

An order of magnitude apart, so ARIn ≈ ASD/√2 and the "BNNs are more robust overall"
headline rests almost entirely on softmax difference. This is Reviewer #1's fourth major
item, stated from the other direction.

Two defensible resolutions:

1. Normalise each criterion to comparable scale before the RMS (the TOPSIS-consistent route).
2. State explicitly that ARIn is deliberately softmax-weighted, and defend why prediction
   confidence deserves the greater weight in an SEU reliability setting.

Either is fine. Silently combining unnormalised criteria of different scale is not.

### Dependency

`docs/analysis/seu_metric_noise_floor.md` records a double-softmax bug affecting the DNN
softmax difference, and `CLAUDE.md` marks the BNN-vs-DNN comparison as blocked on it. Since
ARIn is dominated by the softmax term, that fix and the ARIn justification are the same
piece of work — resolve them together.

---

## 4. Bibliography issues

| Issue | Detail |
|---|---|
| Carbone et al. cited as preprint | Cited as `CoRR abs/2002.04359` in both `PutawaraRevalda_mscthesis.bib` and the ICAART reference list. It is **NeurIPS 2020** (`proceedings.neurips.cc/paper/2020/hash/b3f61131b6eceeb2b14835fa648a48ff`). Citing the paper's primary metric source as a preprint reads as careless. |
| Su et al. cited as preprint | Cited as `CoRR abs/1808.01688`. It is **ECCV 2018**, doi `10.1007/978-3-030-01258-8_39`. |
| Feng et al. cited as preprint | See §2 — published in TMLR 2024. |
| Naming inconsistency | ICAART p.6 introduces "Absolute Accuracy Delta (**AAD**)"; the equation immediately below is labelled "Absolute Accuracy **Difference**"; results tables use "**AAAD**". Three names for two quantities across four pages. Fix: AAD = per-injection value, AAAD = grid average, stated once. |
| Empty DOI | `essbai2024a` has `doi={}`. |
| Duplicate bib entries | `dennispoperobustcnn`, `yan_2020a`, and `Giuffrida2020-ry` each appear twice in `PutawaraRevalda_mscthesis.bib`. BibTeX warns; some styles silently take the last occurrence. |

---

## Priority

1. **Fix Eq. (9)** to express weight perturbation rather than input perturbation. This is a
   correctness bug, not formatting.
2. **Add Cardelli et al. IJCAI 2019** as the upstream source for Softmax Difference.
3. **Justify ARIn** via TOPSIS/MCDA and address the scale-mismatch objection directly
   (coordinate with the noise-floor fix).
4. **Upgrade venues** — Carbone → NeurIPS 2020, Su → ECCV 2018, Feng → TMLR 2024 — and
   de-duplicate the .bib.

## Verified sources

- Carbone et al. (2020), NeurIPS 2020 — https://proceedings.neurips.cc/paper/2020/file/b3f61131b6eceeb2b14835fa648a48ff-Paper.pdf
- Cardelli et al. (2019), IJCAI 2019 — https://www.ijcai.org/proceedings/2019/0789.pdf
- Su et al. (2018), ECCV 2018 — https://link.springer.com/chapter/10.1007/978-3-030-01258-8_39
- Feng et al. (2024), TMLR 2024 — https://arxiv.org/abs/2404.19640
- Pang et al. (2021), arXiv preprint — https://arxiv.org/abs/2106.09223
- Dennis & Pope (2025), ICAART 2025 — doi:10.5220/0013155000003890
- Hwang & Yoon (1981), Springer LNEMS 186 — https://www.semanticscholar.org/paper/a7862f4ac351c8f1caa563b288ac2a63412b6b8f
