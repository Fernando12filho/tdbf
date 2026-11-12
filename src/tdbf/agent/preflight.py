"""Preflight of the procedure agent (design §2, §3.1, §4): no model call.

Before the model sees anything, the controller:
  1. checks the environment and fixes how tests run (`EnvProfile`): the package
     imports; a trivial passing and failing test, placed in a real test directory,
     give pass and fail with the official SWE-bench command, else with today's
     pytest path, else the bug is `environment_invalid`;
  2. extracts the code in the issue (fenced, Trac `{{{ }}}`, indented, `>>>`
     sessions) and its tracebacks, runs each runnable block on the buggy code and
     keeps what it printed (`Snippet`);
  3. ranks location candidates (`Candidates`): code files from traceback frames,
     identifiers looked up in an `ast` index of the repository, and error messages
     found in the source; test files that use those identifiers, the conventional
     neighbour file, the default test directory.

Gold-free: it reads only the issue text, repo, version and base commit
(`PreflightInput`), and runs only on the buggy checkout. The tree is reset at the
end, so the model starts from the pristine checkout.

Pure functions (extraction, parsing, ranking) are separate from the container
steps so `scripts/procedure_preflight_smoke.py` tests them without Docker.
"""

from __future__ import annotations

import ast
import builtins
import json
import keyword
import re
import time
from dataclasses import asdict, dataclass, field

from ..harness.official import (
    DJANGO,
    TestCommand,
    fallback_command,
    official_command,
    parse_result,
)
from ..harness.runner import clean_brt_output, set_brt_module
from .session import ContainerSession
from .tools import RunPythonTool, run_brt_in_session

SNIPPET_TIMEOUT = 90
TEST_TIMEOUT = 180
MAX_SNIPPET_RUNS = 6
MAX_IDENTIFIERS = 15
OUTPUT_KEEP = 4000  # chars of a snippet's output kept (head and tail)

# Top-level package per repository (import check, and which frames are "ours").
TOP_PACKAGE = {
    "astropy/astropy": "astropy",
    "django/django": "django",
    "matplotlib/matplotlib": "matplotlib",
    "mwaskom/seaborn": "seaborn",
    "pallets/flask": "flask",
    "psf/requests": "requests",
    "pydata/xarray": "xarray",
    "pylint-dev/pylint": "pylint",
    "pytest-dev/pytest": "pytest",
    "scikit-learn/scikit-learn": "sklearn",
    "sphinx-doc/sphinx": "sphinx",
    "sympy/sympy": "sympy",
}

# Where a test goes when nothing better is known, in order of preference; the
# first that exists in HEAD wins. Django entries are test packages under tests/.
DEFAULT_TEST_DIRS = {
    "astropy/astropy": ["astropy/tests"],
    "django/django": ["tests/utils_tests", "tests/basic"],
    "matplotlib/matplotlib": ["lib/matplotlib/tests"],
    "mwaskom/seaborn": ["tests", "seaborn/tests"],
    "pallets/flask": ["tests"],
    "psf/requests": ["tests", "."],
    "pydata/xarray": ["xarray/tests"],
    "pylint-dev/pylint": ["tests"],
    "pytest-dev/pytest": ["testing"],
    "scikit-learn/scikit-learn": ["sklearn/tests"],
    "sphinx-doc/sphinx": ["tests"],
    "sympy/sympy": ["sympy/core/tests"],
}

PROBE_FILE = "test_tdbf_probe.py"


@dataclass
class PreflightInput:
    """Everything preflight may know about the bug. Built from the gold-free view;
    there is no field for the gold patch or the hidden tests."""

    instance_id: str
    repo: str
    base_commit: str
    problem_statement: str
    version: str | None

    @classmethod
    def from_instance(cls, inst: object) -> "PreflightInput":
        safe = inst.context_safe_dict()  # type: ignore[attr-defined]
        return cls(safe["instance_id"], safe["repo"], safe["base_commit"],
                   safe["problem_statement"], getattr(inst, "version", None))


# --------------------------------------------------------------------------- #
# Records                                                                      #
# --------------------------------------------------------------------------- #

@dataclass
class EnvProfile:
    repo: str
    version: str | None
    python_version: str | None = None
    top_package: str | None = None
    import_ok: bool | None = None
    command: dict | None = None            # TestCommand.to_json() of the runner chosen
    probe_dir: str | None = None
    probes: list[dict] = field(default_factory=list)   # every probe tried, in order
    test_dirs: list[str] = field(default_factory=list)
    conftests: list[str] = field(default_factory=list)
    default_test_dir: str | None = None
    invalid: bool = False
    reason: str | None = None


@dataclass
class Frame:
    file: str
    line: int | None
    func: str | None
    repo_path: str | None = None   # filled when the file maps to a tracked file


@dataclass
class Traceback:
    frames: list[Frame]
    exc_type: str | None
    message: str


@dataclass
class Snippet:
    index: int
    kind: str          # code | doctest | traceback | shell | text
    source: str        # fenced | trac | indented | inline_doctest
    lang: str | None
    text: str
    script: str | None = None      # what is run (doctests converted)
    runnable: bool = False
    skip_reason: str | None = None
    ran: bool = False
    returncode: int | None = None
    output: str | None = None
    exception: str | None = None   # last exception class in the output
    seconds: float | None = None
    context: str | None = None     # what was prepended on the retry, if any
    first_attempt: dict | None = None  # the run before the retry (NameError)


@dataclass
class Candidate:
    path: str
    score: float
    reasons: list[str]
    line: int | None = None
    symbol: str | None = None
    module: str | None = None      # Django: the test package label


@dataclass
class Candidates:
    identifiers: list[str] = field(default_factory=list)
    code: list[Candidate] = field(default_factory=list)
    tests: list[Candidate] = field(default_factory=list)


@dataclass
class Preflight:
    instance_id: str
    env: EnvProfile
    snippets: list[Snippet]
    tracebacks: list[Traceback]
    candidates: Candidates
    seconds: dict[str, float] = field(default_factory=dict)
    tree_dirty_after: list[str] = field(default_factory=list)

    def to_json(self) -> dict:
        return asdict(self)

    def runner(self) -> TestCommand | None:
        """The chosen test command as an object (None when the env is invalid)."""
        c = self.env.command
        if not c:
            return None
        return TestCommand(c["repo"], c["version"], c["base"], c["source"], c["framework"],
                           c["exports"], c["setup"], c["changes"])


# --------------------------------------------------------------------------- #
# Pure: code blocks in the issue                                               #
# --------------------------------------------------------------------------- #

_FENCE_RE = re.compile(r"^[ \t]*(```|~~~)[ \t]*([\w+-]*)[^\n]*\n(.*?)^[ \t]*\1[ \t]*$",
                       re.MULTILINE | re.DOTALL)
_TRAC_RE = re.compile(r"\{\{\{[ \t]*\n?(?:#!([\w+-]+)[^\n]*\n)?(.*?)\}\}\}", re.DOTALL)
_TB_HEAD = "Traceback (most recent call last)"
_SHELL_START_RE = re.compile(r"^\s*(\$ |# |pip |conda |python -m |python3? \S+\.py|cd |git |>>>\s*$)")
_PROMPT_RE = re.compile(r"^\s*(>>>|\.\.\.)( |$)")
_IPY_IN_RE = re.compile(r"^\s*In \[\d+\]:\s?")
_IPY_CONT_RE = re.compile(r"^\s*\.{3,}:\s?")
_IPY_OUT_RE = re.compile(r"^\s*Out\[\d+\]:")


def normalise(text: str) -> str:
    """Issue texts mix line endings (GitHub issues often carry \\r\\n)."""
    return (text or "").replace("\r\n", "\n").replace("\r", "\n")


def extract_blocks(issue: str) -> list[tuple[str, str | None, str]]:
    """(source, lang, text) for every code-like block, in order of appearance.

    Fenced (``` / ~~~) and Trac ({{{ }}}) blocks first, then 4-space-indented
    blocks and loose `>>>` sessions in the remaining prose."""
    issue = normalise(issue)
    spans: list[tuple[int, str, str | None, str]] = []
    taken: list[tuple[int, int]] = []
    for m in _FENCE_RE.finditer(issue):
        spans.append((m.start(), "fenced", m.group(2) or None, m.group(3)))
        taken.append(m.span())
    for m in _TRAC_RE.finditer(issue):
        if any(a <= m.start() < b for a, b in taken):
            continue
        spans.append((m.start(), "trac", m.group(1), m.group(2)))
        taken.append(m.span())
    # prose left over: indented blocks and inline doctest sessions
    rest = list(issue)
    for a, b in taken:
        rest[a:b] = [" " if ch != "\n" else ch for ch in rest[a:b]]
    prose = "".join(rest)
    lines = prose.split("\n")
    offs, pos = [], 0
    for ln in lines:
        offs.append(pos)
        pos += len(ln) + 1
    i = 0
    while i < len(lines):
        ln = lines[i]
        if ln.startswith(("    ", "\t")) and ln.strip():
            j = i
            while j < len(lines) and (lines[j].startswith(("    ", "\t")) or not lines[j].strip()):
                j += 1
            block = [x[4:] if x.startswith("    ") else x.lstrip("\t") for x in lines[i:j]]
            text = "\n".join(block).strip("\n")
            if text.count("\n") >= 1:
                spans.append((offs[i], "indented", None, text))
            i = j
            continue
        if _PROMPT_RE.match(ln) and ln.strip().startswith(">>>"):
            j = i
            while j < len(lines) and lines[j].strip():
                j += 1
            spans.append((offs[i], "inline_doctest", None, "\n".join(lines[i:j])))
            i = j
            continue
        i += 1
    spans.sort(key=lambda s: s[0])
    seen, out = set(), []
    for _, src, lang, text in spans:
        key = text.strip()
        if key and key not in seen:
            seen.add(key)
            out.append((src, lang, text))
    return out


def doctest_to_script(text: str) -> str | None:
    """Turn a `>>>` (or IPython `In [n]:`) session into a script that echoes each
    expression's value the way the interpreter did. Output lines are dropped."""
    chunks: list[str] = []
    cur: list[str] | None = None
    for ln in text.splitlines():
        s = ln.strip()
        if s.startswith(">>>"):
            if cur is not None:
                chunks.append("\n".join(cur))
            cur = [ln.split(">>>", 1)[1][1:] if ln.split(">>>", 1)[1].startswith(" ")
                   else ln.split(">>>", 1)[1]]
        elif _IPY_IN_RE.match(ln):
            if cur is not None:
                chunks.append("\n".join(cur))
            cur = [_IPY_IN_RE.sub("", ln, count=1)]
        elif cur is not None and (s.startswith("...") and _PROMPT_RE.match(ln)):
            body = ln.split("...", 1)[1]
            cur.append(body[1:] if body.startswith(" ") else body)
        elif cur is not None and _IPY_CONT_RE.match(ln):
            cur.append(_IPY_CONT_RE.sub("", ln, count=1))
        else:
            if cur is not None:
                chunks.append("\n".join(cur))
                cur = None
    if cur is not None:
        chunks.append("\n".join(cur))
    chunks = [c for c in chunks if c.strip()]
    if not chunks:
        return None
    lines = ["import sys as _tdbf_sys",
             "_tdbf_sys.displayhook = lambda v: (v is not None) and print(repr(v))",
             "_tdbf_g = globals()  # names from prepended context stay visible"]
    for c in chunks:
        lines.append(f"exec(compile({c!r} + '\\n', '<issue>', 'single'), _tdbf_g)")
    return "\n".join(lines) + "\n"


# Any length here: a minimum would skip `0` and pair the wrong backticks after it.
_INLINE_RE = re.compile(r"``([^`\n]+?)``|`([^`\n]+?)`")


def inline_expressions(issue: str) -> list[str]:
    """Backticked spans that are a Python expression doing something (an operator
    or a call): `0**-oo`, `f(x).subs(x, 1)`. A bare name or a literal is prose."""
    out: list[str] = []
    for m in _INLINE_RE.finditer(normalise(issue)):
        span = (m.group(1) or m.group(2) or "").strip()
        if not 3 <= len(span) <= 80:
            continue
        try:
            node = ast.parse(span, mode="eval").body
        except (SyntaxError, ValueError):
            continue
        if isinstance(node, (ast.Name, ast.Constant, ast.Attribute)):
            continue
        if not any(isinstance(n, (ast.BinOp, ast.UnaryOp, ast.Call, ast.Compare, ast.Subscript))
                   for n in ast.walk(node)):
            continue
        if span not in out:
            out.append(span)
    return out[:10]


def inline_script(exprs: list[str]) -> str:
    """Evaluate each expression on its own and print `expr -> repr` or the exception."""
    lines = []
    for e in exprs:
        lines += ["try:",
                  f"    print({e + ' -> '!r} + repr(eval({e!r})))",
                  "except Exception as _tdbf_e:",
                  f"    print({e + ' -> raised '!r} + type(_tdbf_e).__name__ + ': ' + str(_tdbf_e)[:200])"]
    return "\n".join(lines) + "\n"


def _parses(code: str) -> bool:
    try:
        ast.parse(code)
        return True
    except (SyntaxError, ValueError):
        return False


def classify_block(text: str, lang: str | None) -> tuple[str, str | None, str | None]:
    """(kind, script, skip_reason). kind: traceback | doctest | code | shell | text.

    A session (`>>>` or IPython `In [n]:`) wins over a traceback in the same block:
    its inputs are run, and the traceback is parsed from the issue text anyway."""
    text = normalise(text)
    if any(ln.strip().startswith(">>>") or _IPY_IN_RE.match(ln) for ln in text.splitlines()):
        script = doctest_to_script(text)
        if script:
            return "doctest", script, None
        return "text", None, "doctest with no statements"
    if _TB_HEAD in text or _IPY_FRAME_RE.match(text.strip()):
        return "traceback", None, "traceback (parsed, not run)"
    lang_l = (lang or "").lower()
    if lang_l in {"sh", "bash", "shell", "console", "shell-session", "bat", "cmd", "powershell"}:
        return "shell", None, f"shell block ({lang_l})"
    if lang_l and lang_l not in {"python", "py", "python3", "py3", "pycon", "ipython", "pytb"}:
        return "text", None, f"not python ({lang_l})"
    first = next((ln for ln in text.splitlines() if ln.strip()), "")
    if _SHELL_START_RE.match(first):
        return "shell", None, "shell command"
    if not _parses(text):
        return "text", None, "does not parse as Python"
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError):
        return "text", None, "does not parse as Python"
    if not tree.body:
        return "text", None, "empty"
    # A lone name or literal ("Foo", "1.2.3") is prose that happens to parse.
    if len(tree.body) == 1 and isinstance(tree.body[0], ast.Expr) and isinstance(
            tree.body[0].value, (ast.Name, ast.Constant, ast.Attribute)):
        return "text", None, "single expression, not a program"
    return "code", text, None


# --------------------------------------------------------------------------- #
# Pure: tracebacks                                                             #
# --------------------------------------------------------------------------- #

_FRAME_RE = re.compile(r'^\s*File "([^"]+)", line (\d+)(?:, in (\S+))?')
_PYTEST_FRAME_RE = re.compile(r"^([\w./\\-]+\.py):(\d+): (?:in (\S+)|([A-Za-z_][\w.]*))")
_IPY_FRAME_RE = re.compile(r"^\s*([~\w./\\-]+\.(?:py|pyx|pxi)) in (\S+?)\(")
_EXC_LINE_RE = re.compile(r"^([A-Za-z_][\w.]*)(?::\s?(.*))?$")


def _is_exception_name(name: str) -> bool:
    leaf = name.split(".")[-1]
    return (leaf.endswith(("Error", "Exception", "Exit", "Interrupt", "Warning", "DoesNotExist",
                           "Failed", "Invalid", "NotFound", "Iteration"))
            or "." in name and leaf[:1].isupper())


def _frame(line: str) -> tuple[Frame, str | None] | None:
    """A traceback frame in any of the three styles, with the exception name when
    the line also carries one (pytest's last `path:N: ValueError`)."""
    fm = _FRAME_RE.match(line)
    if fm:
        return Frame(fm.group(1), int(fm.group(2)), fm.group(3)), None
    im = _IPY_FRAME_RE.match(line)
    if im:
        return Frame(im.group(1), None, im.group(2)), None
    pm = _PYTEST_FRAME_RE.match(line.strip())
    if pm:
        return Frame(pm.group(1), int(pm.group(2)), pm.group(3)), pm.group(4)
    return None


_IPY_ARROW_RE = re.compile(r"^\s*-+>\s*(\d+)")
_MAX_JUNK = 4  # unindented non-frame lines tolerated inside one traceback


def parse_tracebacks(text: str) -> list[Traceback]:
    """Every Python traceback in `text`: standard (`File "x", line N`), IPython
    (`x.py in func(...)` with `---> N`) and pytest (`x.py:N: in func`).

    A traceback starts at a `Traceback (most recent call last)` line or at a frame,
    and ends at its exception line (`ValueError: message`)."""
    out: list[Traceback] = []
    lines = normalise(text).split("\n")
    n, i = len(lines), 0
    while i < n:
        head = _TB_HEAD in lines[i]
        if not head and _frame(lines[i]) is None:
            i += 1
            continue
        hint = None
        if head:  # IPython: "OptionError     Traceback (most recent call last)"
            pre = lines[i].split(_TB_HEAD)[0].strip()
            if pre and _EXC_LINE_RE.match(pre) and _is_exception_name(pre):
                hint = pre
        frames: list[Frame] = []
        exc, msg, e_line = None, "", None
        junk = 0
        j = i + 1 if head else i
        while j < n:
            raw = lines[j]
            s = raw.strip()
            fr = _frame(raw)
            if fr is not None:
                frames.append(fr[0])
                junk = 0
                j += 1
                if fr[1]:
                    exc = fr[1]
                    break
                continue
            am = _IPY_ARROW_RE.match(raw)
            if am and frames and frames[-1].line is None:
                frames[-1].line = int(am.group(1))
            if s.startswith("E ") and e_line is None:
                e_line = s[1:].strip()
            if not s or raw[:1].isspace() or s.startswith(("^", "~", "-", "E ", "<ipython")):
                j += 1
                continue
            if "During handling" in s or "The above exception" in s:
                j += 1
                continue
            em = _EXC_LINE_RE.match(s)
            if em and (frames or hint) and _is_exception_name(em.group(1)):
                exc, msg = em.group(1), (em.group(2) or "").strip()
                k = j + 1  # multi-line message: following unindented lines
                while k < n and lines[k].strip() and not lines[k][:1].isspace() \
                        and _TB_HEAD not in lines[k] and _frame(lines[k]) is None and k - j < 4:
                    msg += "\n" + lines[k].strip()
                    k += 1
                j = k
                break
            junk += 1
            j += 1
            if junk > _MAX_JUNK:
                break
        if exc and not msg and e_line:  # pytest: message from the "E   X: msg" line
            em = _EXC_LINE_RE.match(e_line)
            if em and em.group(1).split(".")[-1] == exc.split(".")[-1]:
                msg = (em.group(2) or "").strip()
        if frames or exc:
            out.append(Traceback(frames, exc or hint, msg))
        i = max(j, i + 1)
    return out


def map_to_repo(path: str, tracked: set[str]) -> str | None:
    """The tracked repo file a traceback path refers to: the longest suffix of
    `path` that is a tracked file (`/usr/.../site-packages/django/db/models/query.py`
    -> `django/db/models/query.py`)."""
    parts = path.replace("\\", "/").split("/")
    for k in range(len(parts)):
        cand = "/".join(parts[k:])
        if cand in tracked:
            return cand
    return None


def exception_in_output(output: str) -> str | None:
    """The last exception class reported in a snippet's output, if any."""
    tbs = parse_tracebacks(output or "")
    for tb in reversed(tbs):
        if tb.exc_type:
            return tb.exc_type
    return None


# --------------------------------------------------------------------------- #
# Pure: identifiers and messages                                               #
# --------------------------------------------------------------------------- #

_BACKTICK_RE = re.compile(r"`([^`\n]{2,120})`")
_DOTTED_RE = re.compile(r"\b[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)+\b")
_CALL_RE = re.compile(r"\b([A-Za-z_]\w*)\s*\(")
_CAMEL_RE = re.compile(r"\b[A-Z][a-z0-9]+(?:[A-Z][a-z0-9]*)+\w*\b")
_SNAKE_RE = re.compile(r"\b[a-z][a-z0-9]*(?:_[a-z0-9]+)+\b")
_WORD_RE = re.compile(r"[A-Za-z_]\w*")

_STOP = set(keyword.kwlist) | set(dir(builtins)) | {
    "self", "cls", "args", "kwargs", "print", "assert", "True", "False", "None", "import",
    "from", "return", "def", "class", "test", "tests", "foo", "bar", "baz", "obj", "value",
    "result", "data", "np", "pd", "plt", "numpy", "pandas", "os", "sys", "re", "json",
    "array", "self_", "setup", "main", "python", "Python", "Django", "GitHub", "TestCase",
    "objects", "assertEqual", "assertTrue", "assertRaises", "pytest", "raises", "dtype",
    "shape", "append", "format", "items", "keys", "values", "get", "list", "dict", "str",
    "int", "float", "len", "range", "type", "isinstance", "super", "init", "__init__",
}


def _leaf(dotted: str) -> str:
    return dotted.split(".")[-1]


def extract_identifiers(issue: str, code_texts: list[str], top_package: str | None) -> list[str]:
    """Identifiers worth looking up, best first. Dotted names keep their qualifier
    (`Engine.render_to_string`); the index is searched by the last part."""
    score: dict[str, float] = {}

    def add(name: str, w: float) -> None:
        name = name.strip(".")
        if not name:
            return
        parts = name.split(".")
        if top_package and parts[0] == top_package and len(parts) > 1:
            name = ".".join(parts[1:]) if len(parts) > 2 else parts[-1]
        leaf = _leaf(name)
        if len(leaf) < 3 or leaf in _STOP or leaf.startswith("__") or leaf.isupper() and len(leaf) < 4:
            return
        score[name] = score.get(name, 0.0) + w

    for m in _BACKTICK_RE.finditer(issue):
        span = m.group(1)
        for d in _DOTTED_RE.findall(span):
            add(d, 3)
        for c in _CALL_RE.findall(span):
            add(c, 3)
        if _WORD_RE.fullmatch(span.strip()):
            add(span.strip(), 3)
    for code in code_texts:
        try:
            tree = ast.parse(code)
        except (SyntaxError, ValueError):
            tree = None
        if tree is not None:
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module and (
                        not top_package or node.module.split(".")[0] == top_package):
                    for a in node.names:
                        add(a.name, 2)
                elif isinstance(node, ast.Call):
                    f = node.func
                    if isinstance(f, ast.Attribute):
                        add(f.attr, 1.5)
                    elif isinstance(f, ast.Name):
                        add(f.id, 1.5)
                elif isinstance(node, ast.Attribute):
                    add(node.attr, 0.5)
        else:
            for c in _CALL_RE.findall(code):
                add(c, 1)
    prose = _BACKTICK_RE.sub(" ", issue)
    for c in _CAMEL_RE.findall(prose):
        add(c, 1)
    for s in _SNAKE_RE.findall(prose):
        add(s, 1)
    for d in _DOTTED_RE.findall(prose):
        if not re.search(r"\.(py|txt|html|org|com|io|rst|cfg)$", d) and d.count(".") <= 4:
            add(d, 1)
    ranked = sorted(score.items(), key=lambda kv: (-kv[1], kv[0]))
    out, leaves = [], set()
    for name, _ in ranked:
        if _leaf(name) in leaves:
            continue
        leaves.add(_leaf(name))
        out.append(name)
    return out[:MAX_IDENTIFIERS]


def message_fragments(tracebacks: list[Traceback], min_len: int = 12) -> list[str]:
    """Literal fragments of the exception messages worth grepping for in the
    source: the longest run of text between quotes/numbers in each message."""
    out = []
    for tb in tracebacks:
        msg = (tb.message or "").split("\n")[0]
        chunks = [c.strip(" :,.") for c in re.split(r"""['"`]|\d+|<[^>]*>|\{[^}]*\}""", msg)]
        chunks = [c for c in chunks if len(c) >= min_len]
        if chunks:
            best = max(chunks, key=len)
            if best not in out:
                out.append(best)
    return out[:5]


# --------------------------------------------------------------------------- #
# Pure: ranking                                                                #
# --------------------------------------------------------------------------- #

def is_test_path(path: str) -> bool:
    p = "/" + path
    name = path.rsplit("/", 1)[-1]
    return ("/tests/" in p or "/testing/" in p or "/test/" in p or name.startswith("test_")
            or name.endswith("_test.py") or name == "conftest.py")


_AUX_DIRS = ("examples/", "example/", "doc/", "docs/", "benchmarks/", "benchmark/", "asv_bench/",
             "tutorials/", "galleries/", "tools/", "ci/", "build_tools/", "extern/", "bin/")


def is_aux_path(path: str) -> bool:
    """Examples, documentation, benchmarks and tooling: never where a bug lives."""
    return path.startswith(_AUX_DIRS)


def rank_code_candidates(tracebacks: list[Traceback],
                         definitions: dict[str, list[dict]],
                         message_hits: dict[str, list[str]],
                         identifiers: list[str],
                         observed: list[Traceback] | None = None) -> list[Candidate]:
    """Code files ranked by evidence. Weights:
      issue traceback: deepest repository frame 5, other repository frames 2;
      traceback of preflight's own run of the issue code: deepest 4, others 1.5;
      definition of an identifier: 3 for the top 3 identifiers else 2 (+1 when the
        qualifier matches the class), divided by sqrt(number of files defining it),
        once per file and identifier (`fit` is defined everywhere and says little);
      an exception message found in the file: 4."""
    cands: dict[str, Candidate] = {}

    def bump(path: str, w: float, why: str, line: int | None = None, sym: str | None = None):
        if is_test_path(path) or is_aux_path(path):
            return
        c = cands.setdefault(path, Candidate(path, 0.0, []))
        c.score += w
        if why not in c.reasons:
            c.reasons.append(why)
        if line is not None and c.line is None:
            c.line, c.symbol = line, sym

    for t, tb in enumerate(tracebacks):
        own = [f for f in tb.frames if f.repo_path]
        for k, f in enumerate(own):
            deepest = k == len(own) - 1
            bump(f.repo_path, 5 if deepest else 2,
                 f"traceback {t + 1} {'deepest frame' if deepest else 'frame'} "
                 f"({f.func or '?'}:{f.line})", f.line if deepest else None, f.func)
    for t, tb in enumerate(observed or []):
        own = [f for f in tb.frames if f.repo_path]
        for k, f in enumerate(own):
            deepest = k == len(own) - 1
            bump(f.repo_path, 4 if deepest else 1.5,
                 f"run of issue code {'deepest frame' if deepest else 'frame'} "
                 f"({f.func or '?'}:{f.line})", f.line if deepest else None, f.func)
    for rank, ident in enumerate(identifiers):
        leaf = _leaf(ident)
        qual = ident.split(".")[-2] if "." in ident else None
        by_file: dict[str, list[dict]] = {}
        for d in definitions.get(leaf, []):
            if not (is_test_path(d["file"]) or is_aux_path(d["file"])):
                by_file.setdefault(d["file"], []).append(d)
        spread = max(1, len(by_file)) ** 0.5
        for path, ds in by_file.items():
            w = 3 if rank < 3 else 2
            match = [d for d in ds if qual and d.get("qualname", "").split(".")[0] == qual]
            if match:
                w += 1
            d = (match or ds)[0]
            bump(path, w / spread, f"defines {d.get('qualname', leaf)}", d.get("line"),
                 d.get("qualname", leaf))
    for frag, files in message_hits.items():
        for f in files[:5]:
            bump(f, 4, f"contains message {frag[:40]!r}")
    return sorted(cands.values(), key=lambda c: (-c.score, c.path))[:10]


def neighbour_tests(repo: str, code_path: str, tracked: set[str]) -> list[str]:
    """Conventional test files for a code file that exist in HEAD."""
    parts = code_path.split("/")
    stem = parts[-1][:-3] if parts[-1].endswith(".py") else parts[-1]
    d = parts[:-1]
    cands = []
    if repo == DJANGO:
        # django/utils/text.py -> tests/utils_tests/test_text.py, tests/text/ ...
        names = [p for p in d[1:] if p not in ("contrib", "db", "core")] or d[1:]
        for n in reversed(names):
            for pkg in (f"{n}_tests", n, f"{n}s"):
                cands += [f"tests/{pkg}/test_{stem}.py", f"tests/{pkg}/tests.py"]
    else:
        for k in range(len(d), 0, -1):
            base = "/".join(d[:k])
            cands += [f"{base}/tests/test_{stem}.py"]
        cands += [f"tests/test_{stem}.py", f"testing/test_{stem}.py", f"test_{stem}.py"]
        if len(d) > 1:
            cands += [f"tests/{'/'.join(d[1:])}/test_{stem}.py"]
    seen, out = set(), []
    for c in cands:
        if c in tracked and c not in seen:
            seen.add(c)
            out.append(c)
    return out


def django_module(path: str) -> str | None:
    """tests/utils_tests/test_text.py -> 'utils_tests' (the test package label)."""
    parts = path.split("/")
    if len(parts) >= 3 and parts[0] == "tests":
        return parts[1]
    if len(parts) == 2 and parts[0] == "tests" and "." not in parts[1]:
        return parts[1]
    return None


def rank_test_candidates(repo: str, code: list[Candidate], usage_hits: dict[str, list[str]],
                         tracked: set[str], default_dir: str | None) -> list[Candidate]:
    """Test files ranked: files that use the issue's identifiers (3 per identifier,
    capped), the conventional neighbour of a top code candidate (4, 2 for lower
    ranks), and the default test directory (1)."""
    cands: dict[str, Candidate] = {}

    def bump(path: str, w: float, why: str):
        c = cands.setdefault(path, Candidate(path, 0.0, []))
        c.score += w
        if why not in c.reasons:
            c.reasons.append(why)

    for ident, files in usage_hits.items():
        if len(files) > 30:  # used everywhere (`fit`): says nothing about where
            continue
        for f in files[:8]:
            bump(f, 3 / max(1, len(files) ** 0.5), f"uses {ident}")
    for rank, cc in enumerate(code[:3]):
        for t in neighbour_tests(repo, cc.path, tracked):
            bump(t, 4 if rank == 0 else 2, f"conventional test of {cc.path}")
    if code:  # tests living in the top code candidate's package
        pkg = code[0].path.rsplit("/", 1)[0] + "/"
        for path in list(cands):
            if path.startswith(pkg):
                bump(path, 2, f"in the package of {code[0].path}")
    if default_dir:
        bump(default_dir + "/", 1, "default test directory")
    out = sorted(cands.values(), key=lambda c: (-c.score, c.path))[:10]
    if repo == DJANGO:
        for c in out:
            c.module = django_module(c.path.rstrip("/"))
    return out


# --------------------------------------------------------------------------- #
# Container steps                                                              #
# --------------------------------------------------------------------------- #

def _keep(text: str, limit: int = OUTPUT_KEEP) -> str:
    """Head and tail of a long output: the assertion/exception is at the end."""
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    half = limit // 2
    return text[:half] + f"\n...[{len(text) - limit} chars omitted]...\n" + text[-half:]


def _probe_code(framework: str, which: str) -> str:
    if framework == "django":
        body = "self.assertEqual(1, 2)" if which == "fail" else "self.assertEqual(1, 1)"
        return ("from django.test import SimpleTestCase\n\n\n"
                "class TdbfEnvProbe(SimpleTestCase):\n"
                f"    def test_tdbf_env_probe_{which}(self):\n"
                f"        {body}\n")
    body = "assert 1 == 2" if which == "fail" else "assert 1 == 1"
    return f"def test_tdbf_env_probe_{which}():\n    {body}\n"


def _git_lines(session: ContainerSession, cmd: str, env: dict | None = None) -> list[str]:
    res = session.exec(cmd, activate=False, env=env, timeout=60)
    if res.returncode != 0:
        return []
    return [ln for ln in res.output.splitlines() if ln.strip()]


def _tracked(session: ContainerSession) -> list[str]:
    """All tracked files (not only .py: IPython frames point at .pyx/.pxi too)."""
    return _git_lines(session, "cd /testbed && git ls-files")


def run_probe(session: ContainerSession, cmd: TestCommand, rel_dir: str) -> dict:
    """Write a trivial passing and a trivial failing test to <rel_dir> and run each
    with `cmd`. ok when they come back pass and fail/assertion."""
    rel = f"{rel_dir.rstrip('/')}/{PROBE_FILE}" if rel_dir not in ("", ".") else PROBE_FILE
    rec: dict = {"runner": cmd.source, "base": cmd.base, "path": rel, "ok": False}
    for which in ("pass", "fail"):
        session.write_file(f"/testbed/{rel}", _probe_code(cmd.framework, which))
        t0 = time.monotonic()
        res = session.exec(cmd.for_file(rel), timeout=TEST_TIMEOUT)
        outcome, reason = parse_result(cmd, res.returncode, res.output)
        rec[which] = {"outcome": outcome, "reason": reason, "returncode": res.returncode,
                      "seconds": round(time.monotonic() - t0, 1),
                      "output_tail": _keep(res.output, 1500)}
    session.exec(f"rm -f /testbed/{rel}", activate=False)
    rec["ok"] = (rec["pass"]["outcome"] == "pass" and rec["fail"]["outcome"] == "fail"
                 and rec["fail"]["reason"] == "assertion")
    return rec


def check_environment(session: ContainerSession, pin: PreflightInput,
                      tracked: list[str]) -> EnvProfile:
    env = EnvProfile(pin.repo, pin.version, top_package=TOP_PACKAGE.get(pin.repo))
    v = session.exec("python --version", timeout=30)
    vm = re.search(r"Python (\d+\.\d+(?:\.\d+)?)", v.output or "")
    env.python_version = vm.group(1) if vm else None
    if env.top_package:
        imp = session.exec(f"python -c 'import {env.top_package}'", timeout=120)
        env.import_ok = imp.returncode == 0
        if not env.import_ok:
            env.probes.append({"import_error": _keep(imp.output, 1500)})
    tset = set(tracked)
    env.test_dirs = sorted({p.rsplit("/", 1)[0] if "/" in p else "." for p in tracked
                            if p.endswith(".py") and is_test_path(p)
                            and p.rsplit("/", 1)[-1].startswith("test")})[:400]
    env.conftests = [p for p in tracked if p.endswith("conftest.py")][:50]
    for d in DEFAULT_TEST_DIRS.get(pin.repo, []):
        if d == "." or any(p.startswith(d + "/") for p in tset):
            env.default_test_dir = d
            break
    if env.default_test_dir is None:
        counts: dict[str, int] = {}
        for p in tracked:
            if is_test_path(p) and "/" in p:
                counts[p.rsplit("/", 1)[0]] = counts.get(p.rsplit("/", 1)[0], 0) + 1
        env.default_test_dir = max(counts, key=counts.get) if counts else "."
    env.probe_dir = env.default_test_dir

    tried: list[TestCommand] = []
    off = official_command(pin.repo, pin.version)
    if off is not None:
        tried.append(off)
    fb = fallback_command(pin.repo)
    if off is None or fb.base != off.base:
        tried.append(fb)
    for cmd in tried:
        for step in cmd.setup:
            r = session.exec(step, timeout=300)
            env.probes.append({"setup": step, "returncode": r.returncode})
        if cmd.framework == "pytest" or cmd.framework == "tox-pytest":
            session.ensure_pytest()
        probe = run_probe(session, cmd, env.probe_dir or ".")
        env.probes.append(probe)
        if probe["ok"]:
            env.command = cmd.to_json()
            break
    if env.command is None:
        env.invalid = True
        env.reason = "no runner gave pass/fail on the trivial probe tests"
    elif env.import_ok is False:
        env.invalid = True
        env.reason = f"import {env.top_package} failed"
    return env


# isympy's documented start-up namespace; sympy issues are written against it.
SYMPY_SESSION = (
    "from sympy import *\n"
    "x, y, z, t = symbols('x y z t')\n"
    "k, m, n = symbols('k m n', integer=True)\n"
    "f, g, h = symbols('f g h', cls=Function)\n"
)


def session_context(repo: str, earlier: list[str]) -> tuple[str, str] | None:
    """(label, code to prepend) for a retry after a NameError, or None."""
    parts, labels = [], []
    if repo == "sympy/sympy":
        parts.append(SYMPY_SESSION)
        labels.append("isympy namespace")
    if earlier:
        parts.append("\n".join(earlier[-3:]) + "\n")
        labels.append(f"previous {min(3, len(earlier))} block(s)")
    if not parts:
        return None
    return " + ".join(labels), "".join(parts)


def run_snippet(session: ContainerSession, repo: str, script: str,
                module: str = "brt_app") -> tuple[int, str]:
    """Run issue code on the buggy checkout: Django through the same probe
    `run_python` uses (runtests.py, contrib apps, test database; `module` is the
    test package whose models are available); other repos as a plain script from a
    scratch directory, so files it writes stay out of /testbed."""
    if repo == DJANGO:
        session.write_file("/tmp/tdbf_probe.py", script)
        result, res = run_brt_in_session(
            session, set_brt_module(RunPythonTool._DJANGO_PROBE, module),
            test_name="test_tdbf_probe", timeout=SNIPPET_TIMEOUT)
        if module == "brt_app":
            session.exec("rm -rf /testbed/tests/brt_app", activate=False)
        else:
            session.exec('rm -f "/testbed/tests/$M/test_tdbf_probe.py"', activate=False,
                         env={"M": module.replace(".", "/")})
        rc = 0 if result.outcome == "pass" else 1
        return rc, clean_brt_output(repo, res.output)
    session.write_file("/tmp/tdbf_snip/snippet.py", script)
    res = session.exec("cd /tmp/tdbf_snip && MPLBACKEND=Agg python snippet.py",
                       timeout=SNIPPET_TIMEOUT)
    return res.returncode, res.output


_INDEX_SCRIPT = r'''
import ast, json, subprocess, sys
wanted = set(json.load(open("/tmp/tdbf_idents.json")))
files = subprocess.check_output(["git", "ls-files", "*.py"], cwd="/testbed").decode().split()
out = {}
AUX = AUX_DIRS
def is_test(p):
    q = "/" + p
    n = p.rsplit("/", 1)[-1]
    return ("/tests/" in q or "/testing/" in q or n.startswith("test_") or n == "conftest.py"
            or p.startswith(AUX))
for f in files:
    if is_test(f):
        continue
    try:
        tree = ast.parse(open("/testbed/" + f, "rb").read())
    except Exception:
        continue
    for node in tree.body:
        kinds = (ast.ClassDef, ast.FunctionDef) + ((ast.AsyncFunctionDef,) if hasattr(ast, "AsyncFunctionDef") else ())
        if isinstance(node, kinds):
            if node.name in wanted:
                out.setdefault(node.name, []).append({"file": f, "line": node.lineno, "qualname": node.name})
            if isinstance(node, ast.ClassDef):
                for sub in node.body:
                    if isinstance(sub, kinds) and sub.name in wanted:
                        out.setdefault(sub.name, []).append(
                            {"file": f, "line": sub.lineno, "qualname": node.name + "." + sub.name})
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id in wanted:
                    out.setdefault(t.id, []).append({"file": f, "line": node.lineno, "qualname": t.id})
print(json.dumps(out))
'''


def index_definitions(session: ContainerSession, identifiers: list[str]) -> dict[str, list[dict]]:
    leaves = sorted({_leaf(i) for i in identifiers})
    if not leaves:
        return {}
    session.write_file("/tmp/tdbf_idents.json", json.dumps(leaves))
    session.write_file("/tmp/tdbf_index.py",
                       f"AUX_DIRS = {_AUX_DIRS!r}\n" + _INDEX_SCRIPT)
    res = session.exec("python /tmp/tdbf_index.py", timeout=120)
    if res.returncode != 0:
        return {}
    try:
        return json.loads(res.output.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return {}


def grep_files(session: ContainerSession, needle: str, *, tests: bool, word: bool) -> list[str]:
    flags = "-l -F" + (" -w" if word else "")
    files = _git_lines(session, f'cd /testbed && git grep {flags} -e "$NEEDLE" -- "*.py"',
                       env={"NEEDLE": needle})
    return [f for f in files if is_test_path(f) == tests]


def build_candidates(session: ContainerSession, pin: PreflightInput, tracked: list[str],
                     tracebacks: list[Traceback], code_texts: list[str],
                     default_dir: str | None,
                     observed: list[Traceback] | None = None) -> Candidates:
    tset = set(tracked)
    for tb in list(tracebacks) + list(observed or []):
        for f in tb.frames:
            f.repo_path = map_to_repo(f.file, tset)
    idents = extract_identifiers(pin.problem_statement, code_texts, TOP_PACKAGE.get(pin.repo))
    defs = index_definitions(session, idents)
    msg_hits = {m: grep_files(session, m, tests=False, word=False)
                for m in message_fragments(tracebacks)}
    code = rank_code_candidates(tracebacks, defs, msg_hits, idents, observed)
    usage = {}
    for ident in idents[:6]:
        hits = grep_files(session, _leaf(ident), tests=True, word=True)
        if hits:
            usage[ident] = hits
    tests = rank_test_candidates(pin.repo, code, usage, tset, default_dir)
    return Candidates(idents, code, tests)


def _porcelain(session: ContainerSession) -> set[str]:
    return set(_git_lines(session, "cd /testbed && git status --porcelain --untracked-files=all"))


def restore_tree(session: ContainerSession, before: set[str]) -> list[str]:
    """Undo only what preflight changed. Not `reset_tree()`: some images carry
    deliberate edits to tracked files from SWE-bench's build (sphinx's tox.ini),
    which a checkout of HEAD would undo. Returns the entries that were new."""
    new = sorted(_porcelain(session) - before)
    for entry in new:
        status, path = entry[:2], entry[3:].strip()
        env = {"P": path}
        if status == "??":
            session.exec('cd /testbed && rm -rf -- "$P"', activate=False, env=env)
        else:
            session.exec('cd /testbed && git checkout -- "$P"', activate=False, env=env)
    return new


def run_preflight(session: ContainerSession, inst: object) -> Preflight:
    """All preflight steps on a started session. No model call; gold-free."""
    pin = PreflightInput.from_instance(inst)
    secs: dict[str, float] = {}
    t0 = time.monotonic()
    before = _porcelain(session)
    tracked = _tracked(session)
    env = check_environment(session, pin, tracked)
    secs["environment"] = round(time.monotonic() - t0, 1)

    t0 = time.monotonic()
    snippets: list[Snippet] = []
    observed: list[Traceback] = []
    runs = 0
    earlier: list[str] = []  # runnable scripts of earlier blocks, in order
    for k, (src, lang, text) in enumerate(extract_blocks(pin.problem_statement)):
        kind, script, skip = classify_block(text, lang)
        sn = Snippet(k, kind, src, lang, text, script=script, runnable=script is not None,
                     skip_reason=skip)
        if sn.runnable and env.invalid:
            sn.skip_reason = "environment invalid"
        elif sn.runnable and runs >= MAX_SNIPPET_RUNS:
            sn.skip_reason = f"over the limit of {MAX_SNIPPET_RUNS} runs"
        elif sn.runnable:
            t1 = time.monotonic()
            rc, out = run_snippet(session, pin.repo, script)
            runs += 1
            exc = exception_in_output(out)
            # A block often continues an earlier one, or (sympy) assumes isympy's
            # namespace. On a NameError, retry once with that context; keep both.
            if exc == "NameError" and runs < MAX_SNIPPET_RUNS:
                ctx = session_context(pin.repo, earlier)
                if ctx:
                    rc2, out2 = run_snippet(session, pin.repo, ctx[1] + script)
                    runs += 1
                    sn.first_attempt = {"returncode": rc, "exception": exc, "output": _keep(out, 1500)}
                    sn.context = ctx[0]
                    sn.script = ctx[1] + script
                    rc, out, exc = rc2, out2, exception_in_output(out2)
            sn.ran, sn.returncode, sn.output, sn.exception = True, rc, _keep(out), exc
            sn.seconds = round(time.monotonic() - t1, 1)
            if exc and exc != "NameError":
                observed += [tb for tb in parse_tracebacks(out) if tb.frames]
        if sn.runnable and script:
            earlier.append(script)
        snippets.append(sn)
    # Inline expressions (`0**-oo`): one run for all of them, each evaluated alone.
    exprs = inline_expressions(pin.problem_statement)
    if exprs:
        script = (SYMPY_SESSION if pin.repo == "sympy/sympy" else "") + inline_script(exprs)
        sn = Snippet(len(snippets), "inline", "inline", None, "\n".join(exprs), script=script,
                     runnable=True, context="isympy namespace" if pin.repo == "sympy/sympy" else None)
        if env.invalid:
            sn.skip_reason = "environment invalid"
        else:
            t1 = time.monotonic()
            rc, out = run_snippet(session, pin.repo, script)
            sn.ran, sn.returncode, sn.output = True, rc, _keep(out)
            sn.exception = exception_in_output(out)
            sn.seconds = round(time.monotonic() - t1, 1)
        snippets.append(sn)
    secs["snippets"] = round(time.monotonic() - t0, 1)

    t0 = time.monotonic()
    tracebacks = parse_tracebacks(pin.problem_statement)
    code_texts = [s.text for s in snippets if s.kind in ("code", "doctest")]
    cands = build_candidates(session, pin, tracked, tracebacks, code_texts, env.default_test_dir,
                             observed)
    secs["candidates"] = round(time.monotonic() - t0, 1)

    dirty = restore_tree(session, before)
    session.exec("rm -rf /tmp/tdbf_snip /tmp/tdbf_probe.py /tmp/tdbf_index.py /tmp/tdbf_idents.json",
                 activate=False)
    return Preflight(pin.instance_id, env, snippets, tracebacks, cands, secs, dirty)
