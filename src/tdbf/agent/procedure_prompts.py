"""Prompts of the procedure agent (design §6) and the handoff record as text.

Each phase gets the same short system prompt plus a phase prompt that contains the
handoff record rendered as text, never the earlier transcripts. The soft variant
gets all steps in one prompt (same wording) and all tools at once.
"""

from __future__ import annotations

import re

SYSTEM = """You are one step in a pipeline that writes a bug reproduction test for a GitHub issue in \
{repo} at the buggy commit {base_commit}. The bug is NOT fixed in this checkout. You cannot see or run \
the fixed code. You work only on the current step; a controller checks your result and moves on.
When the step is done, call {finish_tool} exactly once with the requested fields. Quotes must be \
copied verbatim from their source. Never state a value as fact without its source; if you cannot \
find a source, say so and it will be recorded as unknown. That is an acceptable outcome."""

SOFT_SYSTEM = """You are an autonomous agent that writes a bug reproduction test for a GitHub issue in \
{repo} at the buggy commit {base_commit}. The bug is NOT fixed in this checkout. You cannot see or run \
the fixed code. Quotes must be copied verbatim from their source. Never state a value as fact \
without its source; if you cannot find a source, treat it as unknown. When you are done, call \
submit_test exactly once."""

# How tests are written, per runner. Kept short; Locate adds the directory's own style.
STYLE = {
    "pytest": ("Write plain pytest test functions (`def test_...():`) with `assert`. For an expected "
               "exception use `with pytest.raises(...)`."),
    "tox-pytest": ("Write plain pytest test functions (`def test_...():`) with `assert`; the project's "
                   "conftest fixtures in the test directory are available."),
    "sympy": ("The runner is sympy's own `bin/test` (not pytest): write plain functions named "
              "`test_...` with `assert`. For an expected exception use `raises` from "
              "`sympy.testing.pytest` (older versions: `sympy.utilities.pytest`; check which exists). "
              "Do not use pytest fixtures."),
    "django": ("The runner is Django's `tests/runtests.py`, inside the test package you chose: write "
               "`django.test` TestCase / SimpleTestCase classes, import the package's models absolutely "
               "(`from <package>.models import ...`), and do not configure settings or call "
               "django.setup(). A model defined inside the test file gets no database table."),
}

# v1 (C6): run_python on Django already runs under the test runner (preflight.run_snippet).
DJANGO_RUN_HINT = (
    "\nDjango: run_python already runs your snippet inside a TestCase under Django's test runner, "
    "with settings configured and a test database. Do NOT call settings.configure() or django.setup(). "
    "To use models, pass `module` = a test package under tests/ that defines them (e.g. module "
    "\"queries\" and `from queries.models import ...`); a model defined inside the snippet gets no table.")

UNDERSTAND = """STEP: understand the issue.
Read the issue below, and the PREFLIGHT OUTPUT block after it. The preflight block is NOT part of \
the issue: it is what the controller printed when it ran the issue's code on the buggy checkout. Report:
1. observed: what goes wrong (wrong output, exception, crash). For each, quote the text that shows it \
and say where the quote is: `source` "issue" (the issue text) or "preflight_run" (the preflight \
output). Put the concrete value or exception in `value` (e.g. `0`, `raises KeyError`).
2. expected: what the correct behaviour is, ONLY if the ISSUE states or shows it (e.g. "should \
return", "I expected", "Expected behavior:", correct output shown). Quote the issue. Saying something \
is wrong is not an expected behaviour; the preflight output shows buggy behaviour, never expected.
3. unknowns: at most {max_unknowns}, and ONLY expected values the test must assert that the issue does \
not state (e.g. "what should f(0) return?"). Questions about how to write the test (which API to call, \
equality or identity, which test class) are not unknowns: leave them out.
Do not look at the code in this step.

ISSUE:
{issue}

{preflight}"""

REPRODUCE = """STEP: reproduce the bug on the buggy code.
Observed behaviour to reproduce:
{observed}

{preflight}
Write or adapt a short Python script that shows this behaviour, run it with run_python, and finish \
with the run's id and what it printed. If the issue's example already shows it, use it as is. Run code \
early: read the source only to fix a script that did not work. If you cannot reproduce it within your \
steps, finish with an empty run_id and what happened.{django_hint}
Code candidates (from the issue's tracebacks and names): {code_candidates}

ISSUE (for reference):
{issue}"""

PIN = """STEP: establish the correct expected value for each unknown.
Unknowns:
{unknowns}

What is known so far:
{claims}
{repro}
For each unknown, look for evidence in this order: documentation or docstrings; existing tests of \
the same function; a similar case that works correctly today (run it with run_python). Report each \
value with its source type, the verbatim quote, and the file (docs, existing_test) or the run id \
(observed_run, analogy). If you find no evidence, leave it out. A guess without evidence is recorded \
as unknown, not as a value (source "model" is never treated as verified).{django_hint}
Code candidates: {code_candidates}"""

LOCATE = """STEP: choose where the test should live.
Test-file candidates (ranked, with the reasons):
{test_candidates}
Code candidates: {code_candidates}
Pick the directory whose existing tests cover the code involved, read one test there to learn the \
style (pytest functions or TestCase classes, fixtures), and finish with the directory{django}."""

WRITE = """STEP: write the test.
{repro}
Verified evidence (cite these ids):
{verified}
Unknown values -> do NOT assert these: {unknowns}
Rule for unknown values: leave the assertion out and write the value you would have used as a comment \
`# UNVERIFIED: <value> (<why>)`.
Test home: {test_home} ({style_notes})
How to write tests here: {style}
Write one test that triggers the behaviour from the reproduction and asserts the verified expected \
value(s). Run it with run_test. It must fail on this buggy code AT the assertion. If it errors before \
the assertion, fix the test, not the expected value. Write and run the test first; read files only to \
fix a failing run. Finish with the code and, for each assertion line, the id of the evidence it relies \
on (assertion_map). Every assertion must be in the map.

ISSUE (for reference):
{issue}"""

SOFT_TASK = """Write a bug reproduction test for the issue below by following these steps in order. \
You have all tools at all times; nothing checks the steps for you.

1. Understand: list what is observed (with verbatim quotes from the issue or the preflight output), \
what is expected (only if the issue states it, quoted from the issue) and which expected values are \
unknown (at most {max_unknowns}; questions about how to write the test are not unknowns).
2. Reproduce: show the bug on the buggy code with run_python. Run code early; read the source only to \
fix a script that did not work.{django_hint}
3. Pin expected values: for each unknown, find evidence (documentation or docstrings, existing tests of \
the same function, a similar case that works today). A value without evidence stays unknown.
4. Locate: choose the test directory whose existing tests cover the code; learn its style.
5. Write: write one test that asserts only evidenced expected values (unknown values only as \
`# UNVERIFIED: <value>` comments), run it with run_test (pass `test_dir`{django_arg}), and make it fail on \
the buggy code at its assertion. Then call submit_test with the code, the test_dir{django_arg} and the \
assertion map (line -> the evidence it relies on; use ids you give your evidence, e.g. E1).

How to write tests here: {style}

ISSUE:
{issue}

{preflight}

Code candidates: {code_candidates}
Test-file candidates: {test_candidates}"""

# Feedback when a gate sends the run back (design §6), keyed by gate cause.
SENT_BACK = {
    "passed_on_buggy": ("The test passed on the buggy code, so it did not trigger the bug. The "
                        "reproduction printed {repro_output}; make sure the test does the same thing."),
    "no_evidence": "The test fails at line {line}, but that assertion has no verified evidence.",
    "crashed": "The test crashed before its assertion: {message} Fix the test setup.",
    "symptom_mismatch": "{message}",
    "unmapped_assertion": "{message}",
    "bad_evidence_id": "{message}",
}


def bullet(items: list[str], empty: str = "(none)") -> str:
    return "\n".join(f"- {i}" for i in items) if items else empty


def render_candidates(cands: list[dict], n: int = 5) -> str:
    if not cands:
        return "(none found)"
    return "; ".join(f"{c['path']} ({', '.join(c['reasons'][:2])})" for c in cands[:n])


_INLINE_NAMEERROR = re.compile(r" -> raised NameError: ")
# v2 (C11): a warning the environment emits while importing the repository
# (`/testbed/sympy/core/basic.py:3: DeprecationWarning: Using or importing the ABCs ...`)
# followed by its indented source line.
_IMPORT_WARNING = re.compile(r"^\S*/testbed/\S+:\d+: DeprecationWarning: ")

NO_PREFLIGHT = ("(No preflight output: no code from this issue could be run as is. Quote only the "
                "issue.)")
PF_HEAD = ("PREFLIGHT OUTPUT (not part of the issue: the controller ran the issue's code on the buggy "
           "checkout and this is what it printed). Lines in [square brackets] are labels, not output.")
PF_END = "[end of preflight output]"
# Labels of the preflight block (v2) and of v1's block; stripped from a quote before it is checked.
_PF_LABEL = re.compile(r"^\s*(?:\[(?:preflight[^\]]*|end of preflight output)\]|-{3} [^\n]*:"
                       r"|={10,}[^\n]*|PREFLIGHT OUTPUT[^\n]*)\s*$", re.MULTILINE)


def drop_import_warnings(out: str) -> str:
    """Remove import-time DeprecationWarnings and the source line printed under each (C11)."""
    lines, keep, skip_next = (out or "").splitlines(), [], False
    for ln in lines:
        if skip_next:
            skip_next = False
            if ln.startswith((" ", "\t")):
                continue
        if _IMPORT_WARNING.search(ln):
            skip_next = True
            continue
        keep.append(ln)
    return "\n".join(keep)


def strip_pf_labels(quote: str) -> str:
    """A quote copied from the preflight block, without the block's labels (C9)."""
    return _PF_LABEL.sub("", quote or "").strip()


def shown_snippets(pf: dict) -> list[dict]:
    """The preflight runs the model is shown, with their output as shown.
    v1 (C7): inline expressions that only raised NameError (prose like `f(x)` that is
    not code of this repository) are left out, and an inline run with nothing else
    left is dropped. v2 (C11): import-time DeprecationWarnings are left out. Quotes
    from the preflight block are checked against these outputs and code."""
    shown = []
    for s in pf.get("snippets", []):
        if not s.get("ran"):
            continue
        if s.get("kind") == "inline":
            out = s.get("output") or ""
            exprs = [e for e in (s.get("text") or "").splitlines()
                     if f"{e} -> raised NameError: " not in out]
            outs = [ln for ln in out.splitlines() if not _INLINE_NAMEERROR.search(ln)]
            if not exprs or not any(ln.strip() for ln in outs):
                continue
            s = {**s, "text": "\n".join(exprs), "script": None, "output": "\n".join(outs)}
        s = {**s, "output": drop_import_warnings(s.get("output") or "")}
        shown.append(s)
    return shown[:4]


def shown_code(s: dict) -> str:
    return (s.get("script") or s["text"]) if s.get("kind") != "inline" else s["text"]


def render_preflight(pf: dict, limit: int = 1500) -> str:
    """What preflight ran from the issue and what it printed, as a block labelled as
    NOT issue text (v1, C1). v2 (C9): no block at all when nothing ran, and the
    block's own lines are bracketed labels."""
    ran = shown_snippets(pf)
    if not ran:
        return NO_PREFLIGHT
    parts = [PF_HEAD]
    for s in ran:
        out = (s.get("output") or "(no output)")
        if len(out) > limit:
            out = out[:limit // 2] + "\n...\n" + out[-limit // 2:]
        ctx = f", with {s['context']}" if s.get("context") else ""
        parts.append(f"[preflight ran this {s['kind']} block{ctx}]\n{shown_code(s)[:1200]}\n"
                     f"[preflight: it printed]\n{out}")
    parts.append(PF_END)
    return "\n".join(parts)
