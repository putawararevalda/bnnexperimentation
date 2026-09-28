# EuroSAT BNN Training Explainer — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `docs/learn/eurosat-bnn-training.html` — a self-contained, offline-capable HTML page that teaches how this repository's EuroSAT Bayesian CNN training pipeline works, with three interactive canvas demos and inline caveats about the current implementation.

**Architecture:** One HTML file with inline `<style>` and `<script>`. Nine content sections with a sticky nav. Three vanilla-JS canvas demos. Every code excerpt is verbatim from a source file and carries a `data-src="path:start-end"` attribute. A Python verifier script (`tests/verify_docs_page.py`) re-reads those source files and fails if any excerpt has drifted, so the page cannot silently go stale as the codebase changes.

**Tech Stack:** HTML5, CSS (custom properties, no framework), vanilla ES2020 JS, `<canvas>` 2D context. Verifier in Python 3 standard library only, run via `uv run python`.

## Global Constraints

- **Self-contained.** No external requests of any kind: no CDN scripts, no external stylesheets, no web fonts, no remote images. Everything inline. The page must render fully with the network disconnected.
- **No build step.** The file opens by double-click from the filesystem (`file://`). No bundler, no npm, no server.
- **No dependencies.** No plotting library. Demos are hand-drawn on `<canvas>`.
- **Single file exception to the 200-400 line rule.** `~/.claude/rules/coding-style.md` caps files at 200-400 lines; that rule governs Python source modules. This deliverable is deliberately one file because self-containment is a hard requirement of the spec. The cap does not apply to it. It does apply to `tests/verify_docs_page.py`.
- **Every code excerpt is verbatim** from the named source file, unmodified, with a `data-src="relative/path.py:START-END"` attribute using 1-indexed inclusive line numbers matching the `Read` tool's numbering.
- **Every numeric claim** must come from this plan's verified-numbers table or from a file read during implementation. Do not compute plausible-looking numbers from memory.
- **Caveats are inline**, at the concept they affect, in `<aside class="caveat">`, never presented as recommended practice.
- **Accessibility:** respect `prefers-reduced-motion`; all interactive controls are real `<input>`/`<button>` elements with labels; contrast ratio at least 4.5:1 for body text.
- **Scope:** EuroSAT Bayesian training only. No SEU, no ShipsNet, no deterministic baseline, no MLflow.
- **No git operations.** Never run `git add`, `git commit`, `git stash`, or `git checkout`. All work stays as uncommitted changes in the working tree on branch `docs/eurosat-training-explainer`. The controller snapshots files between tasks to produce review diffs.
- **Do not modify any file outside** `docs/learn/eurosat-bnn-training.html` and `tests/verify_docs_page.py`. In particular, never edit a source file under `src/` or `scripts/` to make an excerpt match — the excerpt is what's wrong, not the source.

## Verified numbers

Every one of these was confirmed against the repository. Use these exact values.

| Quantity | Value | Where it comes from |
|---|---|---|
| EuroSAT images total | 27,000 | `datasplit/split_indices_v2.pkl`, verified by loading |
| Train split | 21,600 | same |
| Test split | 5,400 | same |
| Train/test overlap | 0 | same |
| Batch size | 54 | `scripts/train_eurosat.py:175` |
| Batches per epoch | 400 | 21,600 / 54 |
| `obs_scale` with `--scale-likelihood` | 400.0 | `scripts/train_eurosat.py:178` |
| Input image | 3 x 64 x 64 | `src/data/eurosat.py:20` (`Resize((64,64))`) |
| conv1 weight params | 864 | 32 x 3 x 3 x 3 |
| conv1 bias params | 32 | |
| conv2 weight params | 18,432 | 64 x 32 x 3 x 3 |
| conv2 bias params | 64 | |
| fc1 weight params | 163,840 | 10 x (64 x 16 x 16) = 10 x 16,384 |
| fc1 bias params | 10 | |
| Total latent weights | 183,242 | sum of the six rows above |
| AutoNormal variational params | 366,484 | 183,242 x 2 (one `loc`, one `scale` each) |
| MC samples at inference | 10 | `src/training/svi.py:220` default, called as `num_samples=10` at `scripts/train_eurosat.py:220` |
| Priors swept | 3 | gaussian, laplace, uniform |
| Activations swept | 7 | relu, tanh, sigmoid, sinusoidal, relu6, wg, rwg |
| `b` values swept | 3 | 10.0, 1.0, 0.1 |
| Combinations per variant | 63 | 3 x 7 x 3 |
| Default `init_scale` | 0.25 x b | `scripts/train_eurosat.py:85` |
| Default learning rate | 1e-3 | `scripts/train_eurosat.py:168` |
| Weight decay (variant 03) | 1e-4 | `scripts/train_eurosat.py:167` |
| Dropout p (variant 02) | 0.5 | `src/models/bayesian_cnn.py:46` |
| SmartPool threshold (variant 01) | 10.0 | `src/models/bayesian_cnn.py:43` |

## File structure

| File | Responsibility |
|---|---|
| `docs/learn/eurosat-bnn-training.html` | The entire deliverable: markup, styles, prose, code excerpts, three demos. Created in Task 2, extended in Tasks 3-5. |
| `tests/verify_docs_page.py` | Standalone verifier. Checks excerpt fidelity, nav-anchor integrity, self-containment, and required-numbers presence. Created in Task 1, extended in Task 5. Must stay under 200 lines. |

## Excerpt inventory

These are the exact excerpts the page quotes. Line ranges were read from the
files during planning. If a range no longer matches at implementation time, the
verifier will say so — re-read the file and update both the range and the
excerpt rather than editing the quoted text to fit.

| Section | `data-src` | What it shows |
|---|---|---|
| 2 The model | `src/models/bayesian_cnn.py:89-108` | Layer construction with `PyroSample` |
| 2 The model | `src/models/bayesian_cnn.py:128-144` | `forward`, including the `pyro.sample("obs", ...)` block |
| 3 The prior | `src/models/bayesian_cnn.py:117-126` | `_make_prior`, the three families |
| 4 The guide | `scripts/train_eurosat.py:84-92` | `build_guide`, `init_scale = 0.25 * b` |
| 5 The ELBO | `scripts/train_eurosat.py:177-179` | `obs_scale` assignment under `--scale-likelihood` |
| 6 The SVI loop | `src/training/svi.py:122-136` | Batch loop, the inf/nan `continue`, `svi.step` |
| 6 The SVI loop | `src/training/svi.py:142-142` | The `epoch == 1 or epoch % 10 == 0 or epoch == num_epochs` guard |
| 7 Checkpointing | `src/training/svi.py:178-194` | Best-on-train-accuracy save of the three artifacts |
| 7 Checkpointing | `scripts/train_eurosat.py:206-214` | Restoring the best checkpoint, param store included |
| 8 MC-10 | `src/training/svi.py:239-253` | The MC loop: 10 traces, mean logits, argmax |
| 9 Knobs | `scripts/train_eurosat.py:37-42` | `VARIANT_CONFIG` |
| 1/2 Data | `src/data/eurosat.py:35-51` | Dataset construction, split load, loaders |

---

### Task 1: Excerpt-fidelity verifier

Build the test harness first, against a page that does not exist yet. It fails; that is the point.

**Files:**
- Create: `tests/verify_docs_page.py`

**Interfaces:**
- Consumes: nothing.
- Produces: a CLI verifier. `uv run python tests/verify_docs_page.py` exits 0 on success, 1 on failure, printing one line per problem. Later tasks re-run this unchanged command. It defines the contract that excerpt blocks are `<pre data-src="PATH:START-END"><code>...</code></pre>` with HTML-escaped verbatim content, and that the page lives at `docs/learn/eurosat-bnn-training.html`.

- [ ] **Step 1: Write the verifier**

Create `tests/verify_docs_page.py`:

```python
"""Verify docs/learn/eurosat-bnn-training.html against the source it quotes.

Run: uv run python tests/verify_docs_page.py

Checks:
  1. Every <pre data-src="path:start-end"> excerpt matches those exact lines
     of that file, verbatim.
  2. Every in-page nav href="#id" resolves to an element with that id.
  3. The page makes no external requests (no http(s):// in src/href).
  4. Required numeric facts appear somewhere in the page text.
"""
import html
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PAGE = REPO_ROOT / "docs" / "learn" / "eurosat-bnn-training.html"

EXCERPT_RE = re.compile(
    r'<pre[^>]*\bdata-src="([^"]+)"[^>]*>\s*<code[^>]*>(.*?)</code>\s*</pre>',
    re.DOTALL,
)
HREF_RE = re.compile(r'href="#([^"]+)"')
ID_RE = re.compile(r'\bid="([^"]+)"')
EXTERNAL_RE = re.compile(r'(?:src|href)="https?://[^"]*"')

REQUIRED_FACTS = [
    "27,000", "21,600", "5,400", "183,242", "366,484",
    "16,384", "400", "54",
]


def check_excerpts(page_text: str) -> list[str]:
    """Each excerpt must equal the exact source lines it claims to quote."""
    problems: list[str] = []
    found = 0
    for data_src, raw in EXCERPT_RE.findall(page_text):
        found += 1
        if ":" not in data_src:
            problems.append(f"malformed data-src (no line range): {data_src!r}")
            continue
        path_part, _, span = data_src.rpartition(":")
        source_path = REPO_ROOT / path_part
        if not source_path.is_file():
            problems.append(f"{data_src}: source file not found")
            continue
        try:
            start_s, _, end_s = span.partition("-")
            start, end = int(start_s), int(end_s or start_s)
        except ValueError:
            problems.append(f"{data_src}: unparseable line range {span!r}")
            continue

        source_lines = source_path.read_text(encoding="utf-8").splitlines()
        if not (1 <= start <= end <= len(source_lines)):
            problems.append(
                f"{data_src}: range outside file (file has {len(source_lines)} lines)")
            continue

        expected = "\n".join(source_lines[start - 1:end]).rstrip()
        actual = html.unescape(raw).strip("\n").rstrip()
        if actual != expected:
            problems.append(
                f"{data_src}: excerpt does not match source.\n"
                f"    expected first line: {expected.splitlines()[0]!r}\n"
                f"    page first line:     "
                f"{(actual.splitlines() or [''])[0]!r}")
    if found == 0:
        problems.append("no <pre data-src=...> excerpts found in page")
    else:
        print(f"  checked {found} code excerpt(s)")
    return problems


def check_anchors(page_text: str) -> list[str]:
    ids = set(ID_RE.findall(page_text))
    missing = sorted({h for h in HREF_RE.findall(page_text) if h not in ids})
    return [f"nav link #{m} has no matching id" for m in missing]


def check_self_contained(page_text: str) -> list[str]:
    return [f"external reference not allowed: {m}"
            for m in EXTERNAL_RE.findall(page_text)]


def check_facts(page_text: str) -> list[str]:
    return [f"required fact {fact!r} missing from page"
            for fact in REQUIRED_FACTS if fact not in page_text]


def main() -> int:
    if not PAGE.is_file():
        print(f"FAIL: page not found at {PAGE}")
        return 1

    page_text = PAGE.read_text(encoding="utf-8")
    problems: list[str] = []
    for name, check in [
        ("excerpts", check_excerpts),
        ("anchors", check_anchors),
        ("self-contained", check_self_contained),
        ("facts", check_facts),
    ]:
        found = check(page_text)
        print(f"{'FAIL' if found else 'ok  '}  {name}")
        problems.extend(found)

    if problems:
        print(f"\n{len(problems)} problem(s):")
        for p in problems:
            print(f"  - {p}")
        return 1
    print("\nAll checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run python tests/verify_docs_page.py`

Expected: FAIL, printing `FAIL: page not found at ...docs\learn\eurosat-bnn-training.html`, exit code 1. This confirms the verifier detects a missing page rather than passing vacuously.

- [ ] **Step 3: Leave the change uncommitted**

Do **not** run `git commit` or `git add`. This project is executing the plan with all changes left in the working tree; the controller snapshots files between tasks for review. Simply confirm the file exists and stop.

---

### Task 2: Page skeleton, design system, sections 1-2

Produces a page that opens, navigates, and passes the verifier.

**Files:**
- Create: `docs/learn/eurosat-bnn-training.html`

**Interfaces:**
- Consumes: the verifier contract from Task 1 — excerpts as `<pre data-src="PATH:START-END"><code>`, HTML-escaped.
- Produces: the CSS custom properties, layout shell, and class names every later task reuses: `.section` (a `<section id="...">`), `.caveat` (`<aside>`), `.demo` (a demo container), `.excerpt-label` (the `path:line` caption), `.num` (inline highlighted figure). Nav is `<nav id="toc">` with one `<a href="#section-N">` per section.

- [ ] **Step 1: Create the page skeleton with styles and nav**

Create `docs/learn/eurosat-bnn-training.html`. Start from exactly this shell; later steps fill in sections.

```html
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>How the EuroSAT Bayesian CNN trains</title>
<style>
:root {
  --bg: #0f1115;
  --bg-raised: #171a21;
  --bg-code: #12151b;
  --fg: #d7dbe3;
  --fg-dim: #949cab;
  --fg-bright: #f2f4f8;
  --accent: #7dd3a0;
  --accent-dim: #3f6b53;
  --caution: #e8b563;
  --rule: #262b35;
  --mono: ui-monospace, "Cascadia Mono", "SF Mono", Menlo, Consolas, monospace;
  --sans: system-ui, -apple-system, "Segoe UI", sans-serif;
  --measure: 68ch;
}
* { box-sizing: border-box; }
body {
  margin: 0; background: var(--bg); color: var(--fg);
  font: 16px/1.65 var(--sans);
  -webkit-font-smoothing: antialiased;
}
.layout { display: flex; align-items: flex-start; gap: 3rem; max-width: 1180px; margin: 0 auto; padding: 0 1.5rem; }
#toc {
  position: sticky; top: 0; flex: 0 0 15rem; padding: 3rem 0;
  max-height: 100vh; overflow-y: auto;
}
#toc ol { list-style: none; margin: 0; padding: 0; counter-reset: toc; }
#toc a {
  display: block; padding: 0.35rem 0; color: var(--fg-dim);
  text-decoration: none; font-size: 0.875rem; border-left: 2px solid transparent;
  padding-left: 0.75rem; counter-increment: toc;
}
#toc a::before { content: counter(toc) ". "; color: var(--accent-dim); }
#toc a:hover { color: var(--fg-bright); }
#toc a.active { color: var(--accent); border-left-color: var(--accent); }
main { flex: 1 1 auto; padding: 3rem 0 8rem; min-width: 0; }
h1 { font-size: 2.1rem; line-height: 1.2; margin: 0 0 0.5rem; color: var(--fg-bright); letter-spacing: -0.02em; }
.subtitle { color: var(--fg-dim); max-width: var(--measure); margin: 0 0 3rem; }
h2 { font-size: 1.4rem; margin: 3.5rem 0 1rem; color: var(--fg-bright); letter-spacing: -0.01em; }
h3 { font-size: 1.05rem; margin: 2rem 0 0.5rem; color: var(--fg-bright); }
p, li { max-width: var(--measure); }
section { scroll-margin-top: 2rem; }
section + section { border-top: 1px solid var(--rule); padding-top: 0.5rem; }
code { font-family: var(--mono); font-size: 0.9em; background: var(--bg-raised); padding: 0.1em 0.35em; border-radius: 3px; }
pre {
  background: var(--bg-code); border: 1px solid var(--rule); border-radius: 6px;
  padding: 1rem; overflow-x: auto; font-size: 0.82rem; line-height: 1.55;
}
pre code { background: none; padding: 0; font-size: inherit; }
.excerpt-label {
  font-family: var(--mono); font-size: 0.72rem; color: var(--fg-dim);
  margin: 1.5rem 0 0.35rem; display: block; letter-spacing: 0.02em;
}
.num { color: var(--accent); font-family: var(--mono); font-size: 0.95em; }
.caveat {
  border-left: 3px solid var(--caution); background: rgba(232, 181, 99, 0.06);
  padding: 0.9rem 1.1rem; margin: 1.5rem 0; max-width: var(--measure);
  border-radius: 0 4px 4px 0;
}
.caveat strong { color: var(--caution); display: block; margin-bottom: 0.3rem; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.06em; }
.caveat p { margin: 0.4rem 0 0; font-size: 0.94rem; }
.demo {
  background: var(--bg-raised); border: 1px solid var(--rule); border-radius: 8px;
  padding: 1.25rem; margin: 2rem 0; max-width: var(--measure);
}
.demo h3 { margin-top: 0; }
.demo canvas { width: 100%; height: auto; display: block; border-radius: 4px; }
.controls { display: flex; flex-wrap: wrap; gap: 1rem 1.5rem; align-items: center; margin: 1rem 0 0; font-size: 0.9rem; }
.controls label { display: flex; align-items: center; gap: 0.4rem; color: var(--fg-dim); }
button {
  font: inherit; font-size: 0.875rem; background: var(--accent-dim); color: var(--fg-bright);
  border: none; border-radius: 4px; padding: 0.4rem 0.9rem; cursor: pointer;
}
button:hover { background: var(--accent); color: var(--bg); }
button:disabled { opacity: 0.4; cursor: not-allowed; }
.readout { font-family: var(--mono); font-size: 0.85rem; color: var(--fg-dim); margin-top: 0.85rem; }
.readout b { color: var(--accent); font-weight: 600; }
table { border-collapse: collapse; font-size: 0.9rem; margin: 1.5rem 0; width: 100%; max-width: var(--measure); }
th, td { text-align: left; padding: 0.45rem 0.8rem 0.45rem 0; border-bottom: 1px solid var(--rule); vertical-align: top; }
th { color: var(--fg-bright); font-weight: 600; }
td code { font-size: 0.85em; }
.pipeline { font-family: var(--mono); font-size: 0.78rem; color: var(--fg-dim); line-height: 1.9; overflow-x: auto; }
.pipeline b { color: var(--accent); font-weight: 400; }
@media (max-width: 900px) {
  .layout { display: block; padding: 0 1.1rem; }
  #toc { position: static; padding: 1.5rem 0 0; max-height: none; }
  #toc ol { display: flex; flex-wrap: wrap; gap: 0.25rem 0.75rem; }
  #toc a { border-left: none; padding-left: 0; }
  #toc a.active { border-left: none; }
  main { padding-top: 1.5rem; }
}
@media (prefers-reduced-motion: reduce) {
  * { scroll-behavior: auto !important; transition: none !important; animation: none !important; }
}
html { scroll-behavior: smooth; }
</style>
</head>
<body>
<div class="layout">
<nav id="toc" aria-label="Contents">
  <ol>
    <li><a href="#s1">Why Bayesian at all</a></li>
    <li><a href="#s2">The model</a></li>
    <li><a href="#s3">The prior</a></li>
    <li><a href="#s4">The guide</a></li>
    <li><a href="#s5">The ELBO</a></li>
    <li><a href="#s6">The SVI loop</a></li>
    <li><a href="#s7">Checkpointing</a></li>
    <li><a href="#s8">MC-10 inference</a></li>
    <li><a href="#s9">Knobs and artifacts</a></li>
  </ol>
</nav>
<main>
  <h1>How the EuroSAT Bayesian CNN trains</h1>
  <p class="subtitle">A walk through <code>scripts/train_eurosat.py</code> and everything it calls &mdash; the model, the prior, the guide, the ELBO, and the ten-sample Monte&nbsp;Carlo average that turns a distribution over weights into a single prediction.</p>

  <!-- Sections inserted here by later steps -->

</main>
</div>
<script>
// Nav highlighting: mark the section currently in view.
(function () {
  var links = Array.prototype.slice.call(document.querySelectorAll('#toc a'));
  var sections = links.map(function (a) { return document.getElementById(a.hash.slice(1)); });
  var observer = new IntersectionObserver(function (entries) {
    entries.forEach(function (entry) {
      if (!entry.isIntersecting) return;
      var i = sections.indexOf(entry.target);
      if (i < 0) return;
      links.forEach(function (a) { a.classList.remove('active'); });
      links[i].classList.add('active');
    });
  }, { rootMargin: '-10% 0px -80% 0px' });
  sections.forEach(function (s) { if (s) observer.observe(s); });
})();
</script>
</body>
</html>
```

- [ ] **Step 2: Write section 1, "Why Bayesian at all"**

Insert immediately after the `<!-- Sections inserted here -->` comment. Write the prose yourself; it must convey exactly these points, in this order:

1. An ordinary CNN learns one number per weight. This network learns a *distribution* per weight — <span class="num">183,242</span> of them.
2. Consequence: there is no single "the model". Every forward pass draws a fresh set of weights, so the same image can produce slightly different logits each time.
3. Why this project cares: it studies what a single bit-flip in one weight does. A network whose weights are already understood as a spread, rather than a fixed point, gives you a principled way to ask whether a corrupted weight is an outlier. (One sentence only — SEU is out of scope for this page.)
4. What the page covers and what it does not: covers data → model → prior → guide → ELBO → SVI loop → checkpoint → MC-10 evaluation. Does not cover SEU injection, ShipsNet, or the deterministic baseline.

Wrap as `<section id="s1"><h2>1. Why Bayesian at all</h2>...</section>`.

- [ ] **Step 3: Write section 2, "The model"**

`<section id="s2"><h2>2. The model</h2>`. Required content:

A shape pipeline, marked up as `<div class="pipeline">` (use `&rarr;` and `<b>` on the tensor shapes):

```
input 3x64x64  ->  conv1 3->32, k=3, pad=1  ->  32x64x64  ->  activation
  ->  pool 2x2  ->  32x32x32
  ->  conv2 32->64, k=3, pad=1  ->  64x32x32  ->  activation
  ->  pool 2x2  ->  64x16x16
  ->  flatten  ->  16,384
  ->  fc1 16,384->10  ->  10 logits
```

Then the parameter-count table, using the verified numbers:

| Layer | Weights | Biases |
|---|---|---|
| conv1 | 864 | 32 |
| conv2 | 18,432 | 64 |
| fc1 | 163,840 | 10 |
| **Total** | **183,242 latent variables** | |

Then two excerpts. Precede each with `<span class="excerpt-label">src/models/bayesian_cnn.py:89-108</span>` (matching `data-src`), and copy the source lines **verbatim** from the file, HTML-escaping `<`, `>`, and `&`:

- `src/models/bayesian_cnn.py:89-108` — explain after it: `PyroModule[nn.Conv2d]` is a normal Conv2d whose parameters can be replaced by random variables; assigning `PyroSample(...)` to `.weight` means "this is drawn from that distribution, freshly, on each forward pass," not "this is a stored tensor."
- `src/models/bayesian_cnn.py:128-144` — explain after it: the forward pass is ordinary PyTorch until the last block. `pyro.sample("obs", dist.Categorical(logits=logits), obs=y)` is the statement "the observed labels came from a Categorical parameterised by these logits" — it is what connects the network to the data and makes the ELBO computable. When `y is None` (inference), that block is skipped and you just get logits.

Note the `pyro.poutine.scale(scale=self.obs_scale)` wrapper is visible here and forward-reference section 5 for what it does.

Also cover data loading briefly with the `src/data/eurosat.py:35-51` excerpt: 27,000 EuroSAT images, a pre-computed stratified split loaded from a pickle (21,600 train / 5,400 test, zero overlap), resized to 64x64 and normalised with dataset-specific per-channel mean and std. Add a caveat:

```html
<aside class="caveat">
  <strong>Caveat</strong>
  <p>The default split file is <code>datasplit/split_indices_v2.pkl</code>. Version&nbsp;1 (<code>split_indices.pkl</code>) is corrupt &mdash; the train set contained zero SeaLake images and overlapped the test set on 3,910 indices &mdash; and is kept on disk only for provenance. Any result produced against v1 is invalid.</p>
</aside>
```

- [ ] **Step 4: Run the verifier**

Run: `uv run python tests/verify_docs_page.py`

Expected: PASS on excerpts, anchors, and self-contained. The `facts` check will FAIL, listing the numbers not yet on the page (`366,484` at minimum, since the guide section does not exist). That is correct at this stage — confirm the *excerpt* check reports `ok` and that it says `checked 3 code excerpt(s)`. If any excerpt mismatches, re-read the source file and fix the excerpt, never the source.

- [ ] **Step 5: Open the page in a browser and check it renders**

Open `docs/learn/eurosat-bnn-training.html` in Chrome. Confirm: nav links scroll to sections, the nav item highlights as you scroll, no console errors, and the layout collapses sensibly when the window is narrowed below 900px.

- [ ] **Step 6: Leave the change uncommitted**

Do **not** run `git commit` or `git add`. All changes stay in the working tree; the controller snapshots files between tasks for review.

---

### Task 3: Sections 3-5 and demos 1-2

**Files:**
- Modify: `docs/learn/eurosat-bnn-training.html`

**Interfaces:**
- Consumes: `.section`/`.caveat`/`.demo`/`.controls`/`.readout`/`.excerpt-label`/`.num` classes and CSS variables from Task 2.
- Produces: two demo IIFEs in the page's `<script>`, namespaced `demoPrior` and `demoElbo`, each self-contained with no shared globals. Later tasks add `demoMC` alongside them without touching either.

- [ ] **Step 1: Write section 3, "The prior", with the `_make_prior` excerpt**

`<section id="s3"><h2>3. The prior</h2>`. Required content:

The prior is what you believe about a weight before seeing any data. Here it is centred at `mu = 0.0` and spread by `b`, and every one of the 183,242 weights gets the same prior independently.

Excerpt `src/models/bayesian_cnn.py:117-126`, verbatim, labelled. Then explain each family in terms of `b`:

- **Gaussian** — `b` is the standard deviation. Weights concentrate near zero, large values are strongly discouraged.
- **Laplace** — `b` is the scale. Sharper peak at zero and heavier tails than a Gaussian of the same `b`: it prefers weights near exactly zero but is more tolerant of the occasional large one. The sparsity-inducing choice.
- **Uniform** — `b` is a hard half-width; every value in `[-b, +b]` is equally likely and nothing outside is possible.

Explain `.expand(shape).to_event(len(shape))`: expand gives one independent draw per weight in the layer; `to_event` tells Pyro to treat the whole tensor as a single multi-dimensional random variable rather than a batch of independent ones, which is what makes the log-probabilities sum correctly.

State the swept values: `b` in 10.0, 1.0, 0.1. A `b` of 10.0 is an extremely weak prior (a conv weight of 8.0 is unremarkable); 0.1 is a strong pull toward zero.

- [ ] **Step 2: Add Demo 1 (prior shapes) markup**

Inside section 3:

```html
<div class="demo">
  <h3>Demo: what <code>b</code> does to each prior</h3>
  <canvas id="priorCanvas" width="720" height="300" role="img"
          aria-label="Probability density of the selected prior family"></canvas>
  <div class="controls">
    <label><input type="radio" name="priorFam" value="gaussian" checked> Gaussian</label>
    <label><input type="radio" name="priorFam" value="laplace"> Laplace</label>
    <label><input type="radio" name="priorFam" value="uniform"> Uniform</label>
    <label>b
      <input type="range" id="priorB" min="0" max="2" step="1" value="1">
      <b id="priorBOut" class="num">1.0</b>
    </label>
  </div>
  <p class="readout">Each of the <b>183,242</b> weights is drawn from this distribution before the model sees a single image.</p>
</div>
```

- [ ] **Step 3: Add Demo 1 JavaScript**

Append inside the existing `<script>`, after the nav IIFE:

```javascript
// Demo 1 — prior density shapes.
(function demoPrior() {
  var canvas = document.getElementById('priorCanvas');
  if (!canvas) return;
  var ctx = canvas.getContext('2d');
  var W = canvas.width, H = canvas.height, PAD = 34;
  var B_VALUES = [0.1, 1.0, 10.0];
  var slider = document.getElementById('priorB');
  var bOut = document.getElementById('priorBOut');
  var css = getComputedStyle(document.documentElement);
  var ACCENT = css.getPropertyValue('--accent').trim() || '#7dd3a0';
  var RULE = css.getPropertyValue('--rule').trim() || '#262b35';
  var DIM = css.getPropertyValue('--fg-dim').trim() || '#949cab';

  function density(fam, x, b) {
    if (fam === 'gaussian') {
      return Math.exp(-0.5 * (x / b) * (x / b)) / (b * Math.sqrt(2 * Math.PI));
    }
    if (fam === 'laplace') {
      return Math.exp(-Math.abs(x) / b) / (2 * b);
    }
    return Math.abs(x) <= b ? 1 / (2 * b) : 0;  // uniform
  }

  function family() {
    var checked = document.querySelector('input[name="priorFam"]:checked');
    return checked ? checked.value : 'gaussian';
  }

  function draw() {
    var b = B_VALUES[Number(slider.value)];
    var fam = family();
    bOut.textContent = b.toFixed(1);

    // Fixed x window so changing b visibly changes the shape.
    var XMAX = 3, N = 481;
    var xs = [], ys = [], peak = 0;
    for (var i = 0; i < N; i++) {
      var x = -XMAX + (2 * XMAX * i) / (N - 1);
      var y = density(fam, x, b);
      xs.push(x); ys.push(y);
      if (isFinite(y) && y > peak) peak = y;
    }
    // Share a y-scale across families at b=1 so shapes are comparable.
    var yMax = Math.max(peak, 0.05) * 1.12;

    ctx.clearRect(0, 0, W, H);

    // Axes
    ctx.strokeStyle = RULE; ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(PAD, H - PAD); ctx.lineTo(W - PAD, H - PAD);
    ctx.stroke();
    ctx.fillStyle = DIM; ctx.font = '11px ui-monospace, monospace';
    ctx.textAlign = 'center';
    [-3, -2, -1, 0, 1, 2, 3].forEach(function (t) {
      var px = PAD + ((t + XMAX) / (2 * XMAX)) * (W - 2 * PAD);
      ctx.fillText(String(t), px, H - PAD + 16);
    });
    ctx.fillText('weight value', W / 2, H - 6);

    // Density curve
    ctx.strokeStyle = ACCENT; ctx.lineWidth = 2;
    ctx.beginPath();
    for (var j = 0; j < N; j++) {
      var px2 = PAD + ((xs[j] + XMAX) / (2 * XMAX)) * (W - 2 * PAD);
      var py = H - PAD - Math.min(ys[j] / yMax, 1) * (H - 2 * PAD);
      if (j === 0) ctx.moveTo(px2, py); else ctx.lineTo(px2, py);
    }
    ctx.stroke();

    // Fill under the curve
    ctx.lineTo(W - PAD, H - PAD); ctx.lineTo(PAD, H - PAD); ctx.closePath();
    ctx.fillStyle = 'rgba(125, 211, 160, 0.13)';
    ctx.fill();

    // Annotation
    ctx.fillStyle = DIM; ctx.textAlign = 'left'; ctx.font = '12px system-ui, sans-serif';
    var note = fam === 'uniform'
      ? 'Hard walls at ±' + b.toFixed(1) + ' — nothing outside is possible.'
      : fam === 'laplace'
        ? 'Sharp peak at 0, heavy tails: prefers near-zero weights, tolerates rare large ones.'
        : 'Smooth bell: large weights are strongly discouraged.';
    ctx.fillText(note, PAD, 20);
    if (b === 10.0 && fam !== 'uniform') {
      ctx.fillText('At b=10 the curve is nearly flat here — an almost uninformative prior.', PAD, 38);
    }
  }

  slider.addEventListener('input', draw);
  Array.prototype.forEach.call(
    document.querySelectorAll('input[name="priorFam"]'),
    function (r) { r.addEventListener('change', draw); });
  draw();
})();
```

- [ ] **Step 4: Write section 4, "The guide"**

`<section id="s4"><h2>4. The guide</h2>`. Required content:

The prior says what you believed before. The **posterior** is what you should believe after seeing 21,600 images. It is intractable, so you approximate it with a *guide*: a simpler distribution whose parameters are learned. That approximation is the whole of variational inference.

`AutoNormal` places an independent Gaussian on every latent weight, each with its own learned `loc` (centre) and `scale` (spread). So <span class="num">183,242</span> weights become <span class="num">366,484</span> learned numbers. Emphasise: the *prior* family and the *guide* family are separate choices — this codebase matches them (`AutoLaplace` for a Laplace prior, `AutoUniform` for uniform), but they need not match.

Excerpt `scripts/train_eurosat.py:84-92`, verbatim, labelled. Explain `init_scale`: it is how wide each weight's approximate posterior starts. The default is `0.25 * b`, so it inherits the prior's scale — with `b=10.0` every weight starts with a spread of 2.5, which is enormous, and the first epochs are spent shrinking it. `--init-scale 0.01` starts tight instead.

Add:

```html
<aside class="caveat">
  <strong>Caveat</strong>
  <p><code>init_scale</code> defaults to <code>0.25 * b</code>, which ties the starting width of the posterior to the prior width. For <code>b = 10.0</code> that is an initial per-weight spread of 2.5. If a sweep row at <code>b = 10.0</code> trains badly, an over-wide initialisation is a likely cause before the prior itself is blamed.</p>
</aside>
```

- [ ] **Step 5: Write section 5, "The ELBO", with the `obs_scale` excerpt**

`<section id="s5"><h2>5. The ELBO</h2>`. Required content:

Two competing pressures, stated in words first: make the data likely, and stay close to the prior. The ELBO is their sum, and SVI maximises it (Pyro reports the negative, so the printed loss goes down).

Then the formula, as plain HTML — no MathJax, no external font:

```html
<p style="font-family: var(--mono); font-size: 0.95rem; color: var(--fg-bright); margin: 1.25rem 0;">
  ELBO&nbsp;=&nbsp;E<sub>q</sub>[&nbsp;log&nbsp;p(y&nbsp;|&nbsp;x,&nbsp;w)&nbsp;]&nbsp;&minus;&nbsp;KL(&nbsp;q(w)&nbsp;&#8214;&nbsp;p(w)&nbsp;)
</p>
```

Gloss each term: the first is the fit term (how well weights drawn from the guide explain the labels); the second is the complexity term (how far the guide has moved from the prior). `Trace_ELBO(num_particles=1)` estimates the first with a single weight draw per batch — cheap and noisy; `--num-particles` raises it.

Then the minibatch problem: the KL is paid once per step over *all* 183,242 weights, but each step sees only 54 of 21,600 images. Left uncorrected, the KL is effectively 400x over-weighted. `--scale-likelihood` fixes it by scaling the likelihood by `N / batch_size = 400`, which is what the `pyro.poutine.scale` wrapper in `forward` consumes.

Excerpt `scripts/train_eurosat.py:177-179`, verbatim, labelled. Then:

```html
<aside class="caveat">
  <strong>Caveat &mdash; the two datasets optimise different objectives</strong>
  <p><code>--scale-likelihood</code> is <em>off</em> by default, which preserves the original behaviour: <code>obs_scale = 1.0</code> and a KL term over-weighted by roughly 400x relative to a correct minibatch ELBO. The EuroSAT v02 runs were trained with it <em>on</em> (<code>obs_scale = 400.0</code>); ShipsNet folds 1 and 2 were trained with it <em>off</em>. Loss values are therefore not comparable across the two datasets, and neither are conclusions that depend on how hard the prior was pulling.</p>
</aside>
```

- [ ] **Step 6: Add Demo 2 (ELBO tradeoff) markup and JavaScript**

Markup inside section 5:

```html
<div class="demo">
  <h3>Demo: the two terms pulling against each other</h3>
  <canvas id="elboCanvas" width="720" height="260" role="img"
          aria-label="Relative size of the fit term and the KL term"></canvas>
  <div class="controls">
    <label>posterior moved from prior
      <input type="range" id="elboMove" min="0" max="100" step="1" value="35">
    </label>
    <label><input type="checkbox" id="elboScale"> <code>--scale-likelihood</code> (obs_scale 400)</label>
  </div>
  <p class="readout">Illustrative only &mdash; these are not values from a real run. The point is the <em>ratio</em>: with scaling off, the KL term dominates the objective.</p>
</div>
```

JavaScript, appended after `demoPrior`:

```javascript
// Demo 2 — ELBO tradeoff. Illustrative shapes, not real run values.
(function demoElbo() {
  var canvas = document.getElementById('elboCanvas');
  if (!canvas) return;
  var ctx = canvas.getContext('2d');
  var W = canvas.width, H = canvas.height;
  var move = document.getElementById('elboMove');
  var scaled = document.getElementById('elboScale');
  var css = getComputedStyle(document.documentElement);
  var ACCENT = css.getPropertyValue('--accent').trim() || '#7dd3a0';
  var CAUTION = css.getPropertyValue('--caution').trim() || '#e8b563';
  var DIM = css.getPropertyValue('--fg-dim').trim() || '#949cab';
  var BRIGHT = css.getPropertyValue('--fg-bright').trim() || '#f2f4f8';

  function draw() {
    var t = Number(move.value) / 100;          // 0 = at prior, 1 = far from prior
    var obsScale = scaled.checked ? 400 : 1;
    // Fit improves with distance from the prior, saturating; KL grows quadratically.
    var fitRaw = 1 - Math.exp(-3.2 * t);
    var fit = fitRaw * obsScale;
    var kl = 60 * t * t;
    var total = Math.max(fit + kl, 1e-6);

    ctx.clearRect(0, 0, W, H);
    var barX = 130, barW = W - barX - 40, barY = 70, barH = 54;

    var fitW = (fit / total) * barW;
    ctx.fillStyle = ACCENT;
    ctx.fillRect(barX, barY, fitW, barH);
    ctx.fillStyle = CAUTION;
    ctx.fillRect(barX + fitW, barY, barW - fitW, barH);

    ctx.font = '12px system-ui, sans-serif';
    ctx.textAlign = 'right';
    ctx.fillStyle = DIM;
    ctx.fillText('objective', barX - 12, barY + 33);

    ctx.textAlign = 'left';
    ctx.fillStyle = BRIGHT;
    ctx.font = '13px system-ui, sans-serif';
    ctx.fillText('fit term  ' + Math.round((fit / total) * 100) + '%', barX, barY - 14);
    ctx.textAlign = 'right';
    ctx.fillText('KL term  ' + Math.round((kl / total) * 100) + '%', barX + barW, barY - 14);

    ctx.textAlign = 'left';
    ctx.fillStyle = DIM;
    ctx.font = '12px system-ui, sans-serif';
    ctx.fillText('obs_scale = ' + obsScale, barX, barY + barH + 26);
    var verdict = scaled.checked
      ? 'Likelihood weighted to the full 21,600-image dataset — the KL no longer swamps the fit.'
      : 'Likelihood counts only the 54 images in this batch, while the KL covers all 183,242 weights.';
    ctx.fillText(verdict, barX, barY + barH + 48);

    ctx.fillStyle = DIM;
    ctx.fillText(t < 0.08
      ? 'Guide sits on the prior: KL is nearly zero, but the data is unexplained.'
      : t > 0.85
        ? 'Guide far from the prior: data fits well, complexity penalty is large.'
        : 'Somewhere in between — this is what SVI is searching for.',
      barX, 34);
  }

  move.addEventListener('input', draw);
  scaled.addEventListener('change', draw);
  draw();
})();
```

- [ ] **Step 7: Run the verifier**

Run: `uv run python tests/verify_docs_page.py`

Expected: `excerpts`, `anchors`, `self-contained` all `ok`, now reporting `checked 6 code excerpt(s)`. The `facts` check may still fail on numbers belonging to later sections. Fix any excerpt mismatch by re-reading the source file.

- [ ] **Step 8: Check both demos in the browser**

Open the page in Chrome. For Demo 1: switch all three families, move `b` across 0.1 / 1.0 / 10.0, confirm the curve and the annotation change and the uniform case draws hard walls. For Demo 2: drag the slider end to end and toggle the checkbox, confirm the bar proportions and captions change. Console must be clean.

- [ ] **Step 9: Leave the change uncommitted**

Do **not** run `git commit` or `git add`. All changes stay in the working tree; the controller snapshots files between tasks for review.

---

### Task 4: Sections 6-8 and demo 3

**Files:**
- Modify: `docs/learn/eurosat-bnn-training.html`

**Interfaces:**
- Consumes: everything from Tasks 2 and 3.
- Produces: a third demo IIFE named `demoMC`, independent of `demoPrior` and `demoElbo`.

- [ ] **Step 1: Write section 6, "The SVI loop"**

`<section id="s6"><h2>6. The SVI loop</h2>`. Required content:

What one epoch is: 400 batches of 54 images. What one `svi.step(images, labels)` does, in four beats — draw weights from the guide, run the forward pass, compute the ELBO estimate, backpropagate into the guide's `loc` and `scale` parameters (never into the weights themselves, which are samples, not parameters). Optimiser is `ClippedAdam` at `lr=1e-3`.

Excerpt `src/training/svi.py:122-136`, verbatim, labelled. Point out that `svi.evaluate_loss` is called first purely as a guard, and then:

```html
<aside class="caveat">
  <strong>Caveat &mdash; silently dropped batches</strong>
  <p>When the pre-check finds an infinite or NaN loss, the loop <code>continue</code>s. That batch is skipped entirely: no <code>svi.step</code>, no gradient, and it is excluded from the epoch's <code>batches</code> divisor, so the reported average loss looks normal. A run that quietly trains on a fraction of its data is indistinguishable from a healthy one in the loss CSV. It also costs an extra forward pass per batch, since the loss is computed twice.</p>
</aside>
```

Excerpt `src/training/svi.py:142-142`, verbatim, labelled, followed by:

```html
<aside class="caveat">
  <strong>Caveat &mdash; sparse accuracy sampling</strong>
  <p>Training accuracy is measured at epoch 1, every tenth epoch, and the final epoch. Over a 100-epoch run that is 11 measurements. The "best" checkpoint is the best of those 11, not the best of 100 &mdash; and the accuracy curve you plot has 11 points.</p>
</aside>
```

Note also that accuracy here is measured with a **single** weight draw per batch (`pyro.poutine.trace(guide)` once), not the 10-sample average used at test time — so in-training accuracy is noisier and systematically a little worse than the final reported test accuracy.

- [ ] **Step 2: Write section 7, "Checkpointing"**

`<section id="s7"><h2>7. Checkpointing</h2>`. Required content:

Three artifacts are saved whenever accuracy improves, and you need all three:

| Artifact | File pattern | Why |
|---|---|---|
| Model | `model_{act}_{prior}_epoch_best_{ts}.pth` | Layer structure and any non-Bayesian buffers |
| Guide | `guide_{act}_{prior}_epoch_best_{ts}.pth` | The guide module's own state |
| Param store | `param_store_{act}_{prior}_epoch_best_{ts}.pkl` | **The essential one.** Pyro keeps the learned `loc`/`scale` values in a global param store, and the guide is traced through it. Restore only the two `state_dict`s and you are still sampling from the last epoch's posterior. |

Excerpt `src/training/svi.py:178-194`, verbatim, labelled. Then:

```html
<aside class="caveat">
  <strong>Caveat &mdash; "best" means best on training data</strong>
  <p>The comparison is <code>acc &gt; best_acc</code> where <code>acc</code> is <em>training</em> accuracy. This path has no validation split, so the selected checkpoint is the epoch that best fit the data it was trained on. An overfit epoch can be banked as "best", and the test accuracy reported afterwards is then measured on a checkpoint chosen without reference to held-out data.</p>
</aside>
```

Excerpt `scripts/train_eurosat.py:206-214`, verbatim, labelled — the restore. Explain that the `weights_only=False` argument is required because the param store pickles constraint objects that torch >= 2.6 refuses to load under its safer default, and that this same load is what the SEU scripts perform, which is what keeps accuracy tables and SEU baselines referring to one model.

- [ ] **Step 3: Write section 8, "MC-10 inference"**

`<section id="s8"><h2>8. MC-10 inference</h2>`. Required content — this is the section the page exists for, so be explicit:

The model is a distribution, but a prediction has to be a single class. The bridge: draw `S = 10` complete sets of weights from the guide, run the batch through all 10, average the **logits**, then take the argmax of the average.

Excerpt `src/training/svi.py:239-253`, verbatim, labelled. Walk the four lines that matter:

1. `pyro.poutine.trace(guide).get_trace(images)` — draw one full set of weights and record it.
2. `pyro.poutine.replay(model, trace=trace)` — run the model forced to use exactly those weights.
3. `logits_mc.mean(dim=0)` — average across the 10 draws.
4. `torch.argmax(avg_logits, dim=1)` — one class per image.

Then the subtlety, stated plainly: averaging logits is not the same as averaging softmax outputs. Averaging logits is a geometric mean of the probabilities after renormalising; averaging softmax is the arithmetic mean, and is the one that corresponds to the posterior predictive distribution. This code does the former. Both are defensible; they are not identical, and any downstream metric computed on softmax outputs must be clear about which one it started from.

Two caveats:

```html
<aside class="caveat">
  <strong>Caveat &mdash; results are not reproducible run to run</strong>
  <p>The 10 draws are unseeded. Evaluating the same checkpoint twice gives slightly different accuracy each time. This is real Monte&nbsp;Carlo noise, not a bug, but it means a small accuracy difference between two configurations may be nothing at all &mdash; the noise floor has to be measured before differences of that size can be read as signal.</p>
</aside>
<aside class="caveat">
  <strong>Caveat &mdash; S = 10 is a fixed, unvalidated choice</strong>
  <p><code>num_samples=10</code> is a default that was never swept. Whether the average has converged by 10 draws is an empirical question about this posterior's width, and it has not been checked.</p>
</aside>
```

- [ ] **Step 4: Add Demo 3 (MC-10) markup**

```html
<div class="demo">
  <h3>Demo: watching 10 draws settle on an answer</h3>
  <canvas id="mcCanvas" width="720" height="330" role="img"
          aria-label="Individual sampled logits and their running mean across ten draws"></canvas>
  <div class="controls">
    <button id="mcStep">Draw sample</button>
    <button id="mcAll">Draw all 10</button>
    <button id="mcReroll">Reroll</button>
  </div>
  <p class="readout" id="mcReadout">Draws taken: <b>0</b> / 10</p>
</div>
```

- [ ] **Step 5: Add Demo 3 JavaScript**

```javascript
// Demo 3 — MC-10 averaging. Seeded PRNG so a given roll is repeatable.
(function demoMC() {
  var canvas = document.getElementById('mcCanvas');
  if (!canvas) return;
  var ctx = canvas.getContext('2d');
  var W = canvas.width, H = canvas.height;
  var CLASSES = ['AnnualCrop', 'Forest', 'HerbVeg', 'Highway', 'Industrial',
                 'Pasture', 'PermCrop', 'Residential', 'River', 'SeaLake'];
  var TRUE_CLASS = 3;   // Highway — deliberately a contested case
  var RIVAL = 8;        // River — the class it gets confused with
  var css = getComputedStyle(document.documentElement);
  var ACCENT = css.getPropertyValue('--accent').trim() || '#7dd3a0';
  var CAUTION = css.getPropertyValue('--caution').trim() || '#e8b563';
  var DIM = css.getPropertyValue('--fg-dim').trim() || '#949cab';
  var BRIGHT = css.getPropertyValue('--fg-bright').trim() || '#f2f4f8';
  var RULE = css.getPropertyValue('--rule').trim() || '#262b35';

  var seed = 12345, samples = [], shown = 0;

  function rand() {  // mulberry32
    seed |= 0; seed = (seed + 0x6D2B79F5) | 0;
    var t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  }
  function gauss() {
    var u = Math.max(rand(), 1e-9), v = rand();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }

  function generate() {
    // Posterior mean logits: two plausible classes, the rest low.
    var mean = CLASSES.map(function (_, i) {
      if (i === TRUE_CLASS) return 2.4;
      if (i === RIVAL) return 2.1;
      return -0.4 + rand() * 0.8;
    });
    samples = [];
    for (var s = 0; s < 10; s++) {
      samples.push(mean.map(function (m) { return m + gauss() * 1.5; }));
    }
    shown = 0;
  }

  function runningMean(n) {
    return CLASSES.map(function (_, c) {
      var sum = 0;
      for (var s = 0; s < n; s++) sum += samples[s][c];
      return sum / n;
    });
  }
  function argmax(v) {
    var best = 0;
    for (var i = 1; i < v.length; i++) if (v[i] > v[best]) best = i;
    return best;
  }

  function draw() {
    ctx.clearRect(0, 0, W, H);
    var PAD_L = 92, PAD_R = 24, PAD_T = 46, rowH = 20, barTop = PAD_T + 8;
    var plotW = W - PAD_L - PAD_R;
    var LO = -4, HI = 7;
    function xOf(v) { return PAD_L + ((v - LO) / (HI - LO)) * plotW; }

    ctx.font = '12px system-ui, sans-serif';
    ctx.fillStyle = BRIGHT;
    ctx.textAlign = 'left';
    ctx.fillText(shown === 0
      ? 'No draws yet. Each draw is a complete, different set of 183,242 weights.'
      : 'Mean of ' + shown + ' draw' + (shown === 1 ? '' : 's') + ' (bars) vs. each individual draw (ticks).',
      PAD_L - 68, 22);

    var mean = shown > 0 ? runningMean(shown) : null;
    var meanArg = mean ? argmax(mean) : -1;

    for (var c = 0; c < CLASSES.length; c++) {
      var y = barTop + c * rowH;
      ctx.fillStyle = c === meanArg ? BRIGHT : DIM;
      ctx.font = c === meanArg ? '600 11px ui-monospace, monospace' : '11px ui-monospace, monospace';
      ctx.textAlign = 'right';
      ctx.fillText(CLASSES[c], PAD_L - 10, y + 11);

      ctx.strokeStyle = RULE; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(xOf(LO), y + 7.5); ctx.lineTo(xOf(HI), y + 7.5); ctx.stroke();

      // Individual sampled logits as faint ticks
      for (var s = 0; s < shown; s++) {
        var xs = xOf(samples[s][c]);
        ctx.strokeStyle = 'rgba(148, 156, 171, 0.5)'; ctx.lineWidth = 1;
        ctx.beginPath(); ctx.moveTo(xs, y + 2); ctx.lineTo(xs, y + 13); ctx.stroke();
      }
      // Running-mean bar
      if (mean) {
        var x0 = xOf(0), x1 = xOf(mean[c]);
        ctx.fillStyle = c === meanArg ? ACCENT : 'rgba(125, 211, 160, 0.32)';
        ctx.fillRect(Math.min(x0, x1), y + 4, Math.abs(x1 - x0), 8);
      }
    }

    // Zero line
    ctx.strokeStyle = RULE; ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(xOf(0), barTop); ctx.lineTo(xOf(0), barTop + CLASSES.length * rowH);
    ctx.stroke();

    // Verdict
    ctx.textAlign = 'left'; ctx.font = '12px system-ui, sans-serif';
    var yBot = barTop + CLASSES.length * rowH + 22;
    if (shown === 0) {
      ctx.fillStyle = DIM;
      ctx.fillText('Press "Draw sample" to take one Monte Carlo sample.', PAD_L - 68, yBot);
    } else {
      var thisArg = argmax(samples[shown - 1]);
      ctx.fillStyle = DIM;
      ctx.fillText('This draw alone would predict: ' + CLASSES[thisArg], PAD_L - 68, yBot);
      ctx.fillStyle = meanArg === TRUE_CLASS ? ACCENT : CAUTION;
      ctx.fillText('Mean of ' + shown + ' predicts: ' + CLASSES[meanArg]
        + (shown < 10 ? '' : '   ← this is what predict_data returns'), PAD_L - 68, yBot + 20);
    }

    document.getElementById('mcReadout').innerHTML =
      'Draws taken: <b>' + shown + '</b> / 10';
    document.getElementById('mcStep').disabled = shown >= 10;
    document.getElementById('mcAll').disabled = shown >= 10;
  }

  document.getElementById('mcStep').addEventListener('click', function () {
    if (shown < 10) { shown++; draw(); }
  });
  document.getElementById('mcAll').addEventListener('click', function () {
    shown = 10; draw();
  });
  document.getElementById('mcReroll').addEventListener('click', function () {
    generate(); draw();
  });

  generate();
  draw();
})();
```

- [ ] **Step 6: Run the verifier**

Run: `uv run python tests/verify_docs_page.py`

Expected: all four checks `ok`, `checked 10 code excerpt(s)`, exit 0. Every required fact should now be present. If `facts` still fails, the missing number belongs in a section you have written — add it in prose rather than deleting it from the verifier.

- [ ] **Step 7: Check demo 3 in the browser**

Open the page. Step through all 10 draws one at a time and confirm the individual-draw prediction disagrees with the running mean at least sometimes in the first few draws, and that the mean stabilises by 10. Press Reroll and repeat — a different sequence must appear. Confirm both buttons disable at 10 and re-enable after Reroll. Console clean.

- [ ] **Step 8: Leave the change uncommitted**

Do **not** run `git commit` or `git add`. All changes stay in the working tree; the controller snapshots files between tasks for review.

---

### Task 5: Section 9, caveat audit, final verification

**Files:**
- Modify: `docs/learn/eurosat-bnn-training.html`
- Modify: `tests/verify_docs_page.py`

**Interfaces:**
- Consumes: everything above.
- Produces: the finished deliverable and a verifier that additionally asserts all five spec-mandated caveats are present.

- [ ] **Step 1: Write section 9, "Knobs and artifacts"**

`<section id="s9"><h2>9. Knobs and artifacts</h2>`. Required content:

Excerpt `scripts/train_eurosat.py:37-42`, verbatim, labelled — `VARIANT_CONFIG`. Then the variant table:

| Variant | Label | What changes |
|---|---|---|
| 00 | base | Plain `MaxPool2d`, no dropout, no weight decay |
| 01 | smartpool | `SmartPool` replaces MaxPool, threshold 10.0 — a pooled window whose max exceeds the threshold takes the second-largest value instead |
| 02 | dropout | `Dropout(p=0.5)` after conv2, training only |
| 03 | weight_decay | `ClippedAdam` weight decay 1e-4 |

Then the sweep: 3 priors x 7 activations x 3 `b` values = <span class="num">63</span> combinations per variant, all written to `results/eurosat/bayesian/results_eurosat_v02_{variant}/`. Note `--resume` skips combinations already listed in that directory's `completed_runs.txt`, and that the marker file is re-read before every combination so two windows can share one save directory.

Then the artifacts one combination produces, with the real filename patterns:

| File | Contents |
|---|---|
| `model_{act}_{prior}_epoch_best_{ts}.pth` | Model `state_dict` at the best epoch |
| `guide_{act}_{prior}_epoch_best_{ts}.pth` | Guide `state_dict` at the best epoch |
| `param_store_{act}_{prior}_epoch_best_{ts}.pkl` | Pyro param store — the learned `loc`/`scale` |
| `accuracy_results_{act}_{prior}_{ts}.csv` | Train accuracy at each measured epoch (11 rows for a 100-epoch run) |
| `losses_{act}_{prior}_{ts}.csv` | Average ELBO loss per epoch (100 rows) |
| `config_{act}_{prior}_{ts}.json` | Activation, prior, epochs, best accuracy and its epoch, batch size, train size, prior `mu`/`b` |
| `training_results_{act}_{prior}_{ts}.png` | Four-panel plot: loss, accuracy, `loc` stats, `scale` stats |
| `predictions_{act}_{prior}_{ts}_{acc}.csv` | True and predicted label per test image |

Note that `{act}` is the *function's* `__name__`, not the CLI string — so `sinusoidal` is stored as `sin`, and `wg`/`rwg` as `_actWG`/`_actRWG`. This is why `bayesian_cnn.py` carries legacy aliases in its activation map.

Close the section with a short "where to go next" naming `scripts/eval_seu_eurosat.py` as the subject of a future page, one sentence, no detail.

- [ ] **Step 2: Add the caveat check to the verifier**

In `tests/verify_docs_page.py`, add this constant after `REQUIRED_FACTS`:

```python
REQUIRED_CAVEAT_TOPICS = [
    "scale-likelihood",      # 5: KL over-weighting, datasets differ
    "continue",              # 6: silently dropped inf/nan batches
    "tenth epoch",           # 6: sparse accuracy sampling
    "validation split",      # 7: best chosen on training accuracy
    "unseeded",              # 8: MC non-determinism
]
```

Add this function beside the other checks:

```python
def check_caveats(page_text: str) -> list[str]:
    """Every caveat the spec mandates must survive edits to the page."""
    problems = []
    caveats = re.findall(r'<aside class="caveat">(.*?)</aside>',
                         page_text, re.DOTALL)
    if len(caveats) < 5:
        problems.append(f"expected at least 5 caveat asides, found {len(caveats)}")
    blob = " ".join(caveats)
    for topic in REQUIRED_CAVEAT_TOPICS:
        if topic not in blob:
            problems.append(f"no caveat mentions {topic!r}")
    return problems
```

Register it in `main`'s check list, after `("facts", check_facts)`:

```python
        ("caveats", check_caveats),
```

- [ ] **Step 3: Run the full verifier**

Run: `uv run python tests/verify_docs_page.py`

Expected: five `ok` lines (`excerpts`, `anchors`, `self-contained`, `facts`, `caveats`), `checked 11 code excerpt(s)`, `All checks passed.`, exit 0.

- [ ] **Step 4: Prove the verifier actually catches drift**

The verifier is worthless if it passes unconditionally. Prove otherwise.

Edit one character inside any `<pre data-src=...><code>` block in the page (for example change a variable name in the `_make_prior` excerpt), then run:

Run: `uv run python tests/verify_docs_page.py`

Expected: FAIL on `excerpts`, naming that `data-src` and showing the expected vs. page first line, exit code 1.

Then revert the character, re-run, and confirm it returns to `All checks passed.` Do not commit the temporary edit.

- [ ] **Step 5: Final browser pass**

Open `docs/learn/eurosat-bnn-training.html` in Chrome and confirm all of:

- All nine nav links scroll to the right section; the active item tracks scrolling.
- All three demos respond to every control; console has zero errors and zero warnings.
- Narrow the window below 900px: nav becomes a horizontal list, no horizontal page scroll, canvases scale down without clipping.
- Every `<pre>` with long lines scrolls horizontally inside its own box rather than widening the page.
- Caveats are visually distinct from body prose at a glance.

- [ ] **Step 6: Confirm true self-containment**

In Chrome DevTools, open the Network tab, hard-reload the page, and confirm the only entry is the HTML document itself — no fonts, no images, no scripts.

- [ ] **Step 7: Leave the change uncommitted**

Do **not** run `git commit` or `git add`. All changes stay in the working tree.

---

## Self-review

Checked against `docs/superpowers/specs/2026-08-10-eurosat-bnn-training-docs-design.md`:

- **Spec coverage.** All nine content sections map to Tasks 2-5. All three demos map to Tasks 3 and 4. All five inline caveats from the spec's caveat table are assigned to a section and enforced by `check_caveats` in Task 5. The delivery constraints (single file, no build, no dependencies, offline) are in Global Constraints and verified in Task 5 Step 6. The visual-design requirements (dark-first, single accent, monospace for quoted code, `path:line` labels, sticky nav collapsing below 900px, `prefers-reduced-motion`) are all in the Task 2 stylesheet. The verification requirements are Task 1 plus Task 5 Steps 3-6.
- **Placeholders.** None. Every code step carries complete runnable code; prose steps enumerate the exact facts to convey rather than saying "write the section".
- **Type and name consistency.** CSS class names (`caveat`, `demo`, `controls`, `readout`, `excerpt-label`, `num`, `pipeline`), element ids (`s1`-`s9`, `priorCanvas`, `priorB`, `priorBOut`, `elboCanvas`, `elboMove`, `elboScale`, `mcCanvas`, `mcStep`, `mcAll`, `mcReroll`, `mcReadout`) and the `data-src` attribute contract are used identically in the verifier and in every task that references them. The three demo IIFEs share no globals.
- **One deviation from the skill's defaults, deliberate:** there is no pytest cycle, because the deliverable is a static page. The failing-test-first structure is preserved via `tests/verify_docs_page.py`, which is written first (Task 1), fails against a non-existent page, and is proven to catch real drift in Task 5 Step 4.
