"""Deterministic gates of the procedure agent (design §3, principle 3).

Every check here is string matching, parsing or comparison: no model call, no
judgement. A check that would need judgement is not a gate; it becomes a flag.
All functions are pure so `scripts/procedure_smoke.py` tests them without Docker.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field

from .preflight import exception_in_output, normalise

# Errors that mean a script or test is broken, not that the code under test misbehaved.
AUTHORING_ERRORS = frozenset({"ImportError", "ModuleNotFoundError", "NameError", "SyntaxError",
                              "IndentationError", "TabError"})


@dataclass
class GateResult:
    phase: str
    passed: bool
    cause: str                       # short code: "ok", "quote_missing", "passed_on_buggy", ...
    message: str = ""                # what the model is told when it is sent back
    details: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Quotes                                                                       #
# --------------------------------------------------------------------------- #

def _flat(text: str) -> str:
    """Whitespace-collapsed text without markdown backticks or quote styling."""
    t = normalise(text or "").replace("`", "")
    t = t.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
    return re.sub(r"\s+", " ", t).strip()


def quote_in(source: str, quote: str, *, min_len: int = 3) -> bool:
    """Is `quote` verbatim in `source`, up to whitespace and backticks?"""
    q = _flat(quote).strip(" \"'")
    if len(q) < min_len:
        return False
    return q in _flat(source)


# --------------------------------------------------------------------------- #
# Symptom matcher (D5: deterministic only)                                     #
# --------------------------------------------------------------------------- #

_EXC_NAME_RE = re.compile(r"\b((?:[A-Za-z_]\w*\.)*[A-Z]\w*(?:Error|Exception|Warning|Exit|DoesNotExist|"
                          r"NotFound|Invalid\w*))\b")
_LITERAL_RE = re.compile(r"`([^`]+)`|\"([^\"]+)\"|'([^']+)'")


def claim_tokens(claims: list[dict]) -> tuple[set[str], list[str]]:
    """(exception class names, literal value tokens) named by observed claims.

    Literal tokens come from backticked or quoted spans of the claim's value, and
    the value itself when it is short and has no spaces (`0`, `None`, `[1, 2]`)."""
    excs: set[str] = set()
    lits: list[str] = []
    for c in claims:
        value = str(c.get("value") or "")
        for m in _EXC_NAME_RE.finditer(value):
            excs.add(m.group(1).split(".")[-1])
        for m in _LITERAL_RE.finditer(value):
            tok = next(g for g in m.groups() if g is not None).strip()
            if tok and tok not in lits:
                lits.append(tok)
        v = value.strip()
        if v and " " not in v and len(v) <= 40 and v not in lits and not _EXC_NAME_RE.fullmatch(v):
            lits.append(v)
    return excs, lits


def _token_in(output: str, tok: str) -> bool:
    if len(tok) <= 3:
        # `0`, `1`, `[]`: too short to search anywhere (0 is inside `0**-oo`); it must
        # stand as a printed or compared value: after start of line, "->", "=", ":",
        # "assert", and before end of line, a comparison or a comma.
        pat = (rf"(?:^|->|=|:|assert)\s*{re.escape(tok)}\s*(?:$|==|!=|<|>|,|\))")
        return re.search(pat, output, re.MULTILINE) is not None
    if re.fullmatch(r"[\w.]+", tok):  # a word or number: whole-token match
        return re.search(rf"(?<![\w.]){re.escape(tok)}(?![\w])", output) is not None
    return _flat(tok) in _flat(output)


def symptom_match(observed: list[dict], output: str) -> tuple[str, list[str], str]:
    """(status, matched tokens, explanation). status: matched | partial | none | unchecked.

    matched  the run raised an exception class the issue names, or printed a value
             the issue shows (and did not die on an authoring error);
    partial  it raised a different (non-authoring) exception than the one named;
    none     neither;
    unchecked there was nothing concrete in the observed claims to compare."""
    output = output or ""
    exc = exception_in_output(output)
    leaf = exc.split(".")[-1] if exc else None
    if leaf in AUTHORING_ERRORS:
        return "none", [], f"the script itself failed with {leaf}"
    excs, lits = claim_tokens(observed)
    if not excs and not lits:
        return "unchecked", [], "the observed claims name no exception or value to compare"
    if leaf and leaf in excs:
        return "matched", [leaf], f"raised {leaf}, as the issue says"
    found = [t for t in lits if _token_in(output, t)]
    if found and not excs:
        return "matched", found, f"printed {found[0]!r}, as the issue shows"
    if found and excs and not leaf:
        return "partial", found, f"printed {found[0]!r} but did not raise {sorted(excs)}"
    if leaf and excs:
        return "partial", [leaf], f"raised {leaf}, but the issue names {sorted(excs)}"
    if leaf and not excs:
        return "partial", [leaf], f"raised {leaf}; the issue shows a value, not an exception"
    want = sorted(excs) + lits[:3]
    return "none", [], f"the output shows none of {want}"


# --------------------------------------------------------------------------- #
# Test structure                                                               #
# --------------------------------------------------------------------------- #

def failing_line(output: str, test_file: str) -> int | None:
    """The last line of `test_file` named in a traceback (standard, pytest long or
    short style). That is where the test failed."""
    name = re.escape(test_file.rsplit("/", 1)[-1])
    pats = [rf'File "[^"]*{name}", line (\d+)', rf"(?:^|[\s/]){name}:(\d+):"]
    last = None
    pos = -1
    for p in pats:
        for m in re.finditer(p, output or "", re.MULTILINE):
            if m.start() > pos:
                pos, last = m.start(), int(m.group(1))
    return last


def _stmts(tree: ast.AST):
    for node in ast.walk(tree):
        if isinstance(node, ast.stmt):
            yield node


def statement_at(code: str, line: int) -> tuple[int, int] | None:
    """(first, last) line of the innermost simple statement containing `line`."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    best = None
    for s in _stmts(tree):
        end = getattr(s, "end_lineno", s.lineno)
        if s.lineno <= line <= end and not isinstance(
                s, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if best is None or (end - s.lineno) < (best[1] - best[0]):
                best = (s.lineno, end)
    return best


_ASSERT_CALL_RE = re.compile(r"^(assert\w*|fail\w*|raises|warns)$")


def _is_assert_call(node: ast.AST) -> bool:
    if not isinstance(node, ast.Call):
        return False
    f = node.func
    name = f.attr if isinstance(f, ast.Attribute) else f.id if isinstance(f, ast.Name) else ""
    return bool(_ASSERT_CALL_RE.match(name))


def assertion_statements(code: str) -> list[tuple[int, int]]:
    """Line ranges of every statement that asserts: `assert`, `self.assert*(...)`,
    `pytest.raises` / `raises` / `assertRaises` (call or `with` header), `pytest.fail`."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []
    out = []
    for s in _stmts(tree):
        if isinstance(s, ast.Assert):
            out.append((s.lineno, getattr(s, "end_lineno", s.lineno)))
        elif isinstance(s, ast.Expr) and _is_assert_call(s.value):
            out.append((s.lineno, getattr(s, "end_lineno", s.lineno)))
        elif isinstance(s, (ast.With, ast.AsyncWith)) and any(
                _is_assert_call(it.context_expr) for it in s.items):
            out.append((s.lineno, s.lineno))
    return sorted(set(out))


_DETAIL_RES = (
    re.compile(r"^E\s+(?!AssertionError\s*$)\S", re.MULTILINE),      # pytest: E   assert 0 == 1
    re.compile(r"^(?:\w+\.)*AssertionError:\s*\S", re.MULTILINE),     # AssertionError: 0 != zoo
    re.compile(r"^(?:\w+\.)*\w+(?:Error|Exception):\s*\S", re.MULTILINE),  # raised with a message
)


def assertion_detail(output: str) -> bool:
    """Does the failure output show what was compared (or the exception's message)?"""
    return any(p.search(output or "") for p in _DETAIL_RES)


def check_write(code: str, run: dict, amap: list[dict], verified_ids: set[str],
                all_ids: set[str], *, accept_exception: bool,
                symptom_tokens: list[str] | None) -> GateResult:
    """The Write gate (design §3.6). `run` is the controller's own run of `code` on
    the buggy tree: {outcome, reason, output, failing_line}.

    1. it fails, at an assertion (Django: or with an exception from the code);
    2. every assertion is mapped to verified evidence (D2: unverified values may
       only appear as `# UNVERIFIED:` comments), and the failing statement is mapped;
    3. when the reproduction matched the issue on a concrete token, the failure
       output shows that token (the test fails the way the bug shows)."""
    outcome, reason = run.get("outcome"), run.get("reason")
    out = run.get("output") or ""
    if outcome == "pass":
        return GateResult("write", False, "passed_on_buggy",
                          "The test PASSED on the buggy code, so it did not trigger the bug.")
    ok_reasons = {"assertion", "exception"} if accept_exception else {"assertion"}
    if outcome != "fail" or reason not in ok_reasons:
        exc = exception_in_output(out)
        return GateResult("write", False, "crashed",
                          f"The test did not fail at an assertion (outcome={outcome}, "
                          f"failure_reason={reason}{', ' + exc if exc else ''}). "
                          "Fix the test setup, not the expected value.",
                          {"exception": exc})
    mapped: dict[int, str] = {}
    for item in amap or []:
        try:
            mapped[int(item.get("line"))] = str(item.get("evidence_id") or "").strip()
        except (TypeError, ValueError):
            continue
    unknown_ids = sorted({e for e in mapped.values() if e and e not in all_ids})
    if unknown_ids:
        return GateResult("write", False, "bad_evidence_id",
                          f"assertion_map names evidence ids that do not exist: {unknown_ids}. "
                          f"Valid ids: {sorted(all_ids)}.")

    def ids_for(rng: tuple[int, int]) -> list[str]:
        return [e for ln, e in mapped.items() if rng[0] <= ln <= rng[1]]

    unmapped, unverified = [], []
    for rng in assertion_statements(code):
        ids = ids_for(rng)
        if not ids:
            unmapped.append(rng[0])
        elif not any(e in verified_ids for e in ids):
            unverified.append(rng[0])
    if unmapped:
        return GateResult("write", False, "unmapped_assertion",
                          f"The assertion(s) at line(s) {unmapped} are not in assertion_map. Map each "
                          "assertion to the id of the verified evidence it checks, or remove it and "
                          "keep the value as a `# UNVERIFIED:` comment.", {"lines": unmapped})
    fl = run.get("failing_line")
    stmt = statement_at(code, fl) if fl else None
    fail_ids = ids_for(stmt) if stmt else []
    if unverified or not any(e in verified_ids for e in fail_ids):
        lines = unverified or ([fl] if fl else [])
        return GateResult("write", False, "no_evidence",
                          f"The test fails at line {fl}, but that assertion has no verified evidence"
                          + (f" (unverified at lines {unverified})" if unverified else "") + ".",
                          {"failing_line": fl, "lines": lines})
    if symptom_tokens and not assertion_detail(out):
        # sympy's bin/test and plain unittest asserts print a bare AssertionError: the
        # values are not in the output, so the check cannot be made (a flag, not a gate).
        return GateResult("write", True, "ok", "", {"failing_line": fl, "evidence": fail_ids,
                                                    "symptom": "not_checkable"})
    if symptom_tokens:
        if not any(_token_in(out, t) for t in symptom_tokens):
            return GateResult("write", False, "symptom_mismatch",
                              f"The test fails, but its output does not show what the reproduction "
                              f"showed ({symptom_tokens[:3]}), so it may fail for another reason.",
                              {"tokens": symptom_tokens})
    return GateResult("write", True, "ok", "", {"failing_line": fl, "evidence": fail_ids})
