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

REQUIRED_CAVEAT_TOPICS = [
    "scale-likelihood",      # 5: KL over-weighting, datasets differ
    "continue",              # 6: silently dropped inf/nan batches
    "tenth epoch",           # 6: sparse accuracy sampling
    "validation split",      # 7: best chosen on training accuracy
    "unseeded",              # 8: MC non-determinism
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
        ("caveats", check_caveats),
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
