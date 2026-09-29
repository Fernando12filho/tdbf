"""Thin wrapper around `swebench.harness.run_evaluation`.

We do NOT reimplement container provisioning or test scoring. The
official harness already: builds an isolated per-instance Docker image,
checks out base_commit, applies a candidate patch, runs the instance's
FAIL_TO_PASS / PASS_TO_PASS commands, and writes a structured report.

This module's jobs:
  1. Write the predictions.jsonl the harness expects.
  2. Invoke the harness scoped to one or a few instance_ids.
  3. Read back the per-instance report JSON.
  4. Run a generated test patch directly inside the instance container
     (bypassing eval.sh) to get the fail->pass signal for BRT classification.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

# Matches exception class names on pytest "E " traceback lines, e.g.:
#   E   AssertionError: assert 'BEGIN' not in ...
#   E   django.core.exceptions.ImproperlyConfigured: ...
#   E   ImportError: cannot import name 'X'
#   E   django.core.exceptions.EmptyResultSet          (no message, no colon)
#
# The name is deliberately NOT suffix-restricted. An earlier version required
# it to end in Error/Exception/Configured/Warning, which silently matched
# NOTHING for a large family of real crashes — Django's FieldDoesNotExist,
# <Model>.DoesNotExist, MultipleObjectsReturned, EmptyResultSet, and the
# builtins SystemExit / StopIteration — so those parsed as "no evidence" and
# a crashing test was accepted as a reproduction.
_E_LINE_RE = re.compile(r"^E\s+(.*)$", re.MULTILINE)

# "The test reached its assertion and it fired." `assert ...` is pytest's
# rewritten form (it prints no AssertionError line at all); `Failed` is pytest's
# own — `pytest.fail(...)` and `Failed: DID NOT RAISE` from a pytest.raises block
# that saw no exception. All three are assertions about behaviour, not crashes.
_ASSERT_SIGNAL_RE = re.compile(r"^(?:AssertionError\b|Failed\b|assert\s)")

# A leading `Name:` or `Name` on an E-line — a *candidate* exception only.
_EXC_NAME_RE = re.compile(r"^([A-Za-z_][\w.]*)\s*(?::|$)")

# Crash-y builtins whose names carry no Error/Exception suffix.
_CRASH_BUILTIN_NAMES = frozenset(
    {"SystemExit", "StopIteration", "KeyboardInterrupt", "GeneratorExit"}
)


def _looks_like_exception(dotted: str) -> bool:
    """Does this leading `Name:` name a real exception class?

    A plain "any identifier followed by a colon" test is far too loose: an
    assertion *message* is printed on E-lines too, so a multi-line message like

        E   AssertionError: separability_matrix gave wrong result.
        E     Expected:
        E     Got:

    offers up `Expected` and `Got` as exception names. Requiring a dotted path,
    an Error/Exception suffix, or a known suffix-less builtin keeps Django's
    FieldDoesNotExist / DoesNotExist / MultipleObjectsReturned / EmptyResultSet
    and SystemExit / StopIteration while rejecting ordinary prose.
    """
    leaf = dotted.split(".")[-1]
    if leaf.endswith("Warning"):
        return False  # a warning is never why a test failed
    if "." in dotted:
        return True  # module-qualified -> an exception, not prose
    if leaf.endswith(("Error", "Exception")):
        return True
    return leaf in _CRASH_BUILTIN_NAMES

BRT_TEST_FILENAME = "test_brt.py"
BRT_TEST_PATH = BRT_TEST_FILENAME  # backward-compat alias (repo root default)

# Repos whose tests live in a subdirectory that provides the right imports /
# environment automatically (conftest, settings, fixtures).
_REPO_TEST_SUBDIRS: dict[str, str] = {
    "django/django": "tests",
    "django/django-rest-framework": "tests",
}

# Setup preamble injected at the top of the generated test for repos that
# need framework bootstrapping the model shouldn't have to write.
_REPO_PREAMBLES: dict[str, str] = {
    "django/django": (
        "import sys, os, django\n"
        "sys.path.insert(0, '/testbed/tests')\n"
        "os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'test_sqlite')\n"
        "django.setup()\n"
    ),
    "django/django-rest-framework": (
        "import django\n"
        "from django.conf import settings\n"
        "if not settings.configured:\n"
        "    settings.configure(DATABASES={'default': {'ENGINE': 'django.db.backends.sqlite3', 'NAME': ':memory:'}}, INSTALLED_APPS=['rest_framework'])\n"
        "django.setup()\n"
    ),
}


# Repos whose OWN test suite cannot be driven by a bare `python -m pytest <file>`.
# Django's tests need its runner plus a settings module, and they are addressed by
# dotted LABEL, not by path. This is the same invocation the SWE-bench grader uses
# (from a generated eval.sh):
#   ./tests/runtests.py --verbosity 2 --settings=test_sqlite --parallel 1 <label>
# Under plain pytest every Django test instead errors in setUpClass with
# "ImproperlyConfigured: Requested setting DATABASES", which is what the fix agent
# saw on every Django instance -- its regression check returned nothing usable.
# The generated BRT is unaffected: it carries _REPO_PREAMBLES and runs under pytest.
_REPO_TEST_RUNNERS: dict[str, str] = {
    "django/django": (
        './tests/runtests.py --verbosity 1 --settings=test_sqlite --parallel 1 "$TLABEL"'
    ),
}


def brt_test_path(repo: str) -> str:
    """Return the repo-relative path where the BRT file should be placed."""
    subdir = _REPO_TEST_SUBDIRS.get(repo, "")
    return f"{subdir}/{BRT_TEST_FILENAME}" if subdir else BRT_TEST_FILENAME


def repo_test_label(repo: str, path: str) -> str | None:
    """Translate a repo-relative test FILE path into the label that repo's own
    runner expects. None when the repo has no custom runner (pytest takes the
    path as-is).

        django/django, "tests/check_framework/test_templates.py"
            -> "check_framework.test_templates"

    The agent keeps giving us a path (one interface for every repo); the
    translation happens here.
    """
    if repo not in _REPO_TEST_RUNNERS:
        return None
    p = path.strip().strip("/")
    subdir = _REPO_TEST_SUBDIRS.get(repo, "")
    if subdir and p.startswith(f"{subdir}/"):
        p = p[len(subdir) + 1:]
    if p.endswith(".py"):
        p = p[:-3]
    return p.replace("/", ".") or None


def repo_test_command(repo: str) -> tuple[str, bool]:
    """Shell command for running an EXISTING test file from the repo, and whether
    it expects a label ("$TLABEL") rather than a path ("$TPATH").

    Both forms read the untrusted, model-supplied argument from the environment,
    never from interpolated command text (see ContainerSession.exec).
    """
    custom = _REPO_TEST_RUNNERS.get(repo)
    if custom is not None:
        return custom, True
    return 'python -m pytest "$TPATH" --tb=short -q', False


def inject_setup_preamble(repo: str, test_code: str) -> str:
    """Prepend framework setup boilerplate so the model doesn't have to."""
    preamble = _REPO_PREAMBLES.get(repo, "")
    return preamble + test_code if preamble else test_code


def fetch_python_version(instance: "object") -> str | None:
    """Return the Python minor version (e.g. '3.6') from the instance's testbed env."""
    instance_id: str = instance.instance_id  # type: ignore[attr-defined]
    image_name = f"sweb.eval.x86_64.{instance_id}:latest"
    proc = subprocess.run(
        ["docker", "run", "--rm", image_name,
         "bash", "-c",
         "source /opt/miniconda3/bin/activate && conda activate testbed && python --version"],
        capture_output=True, text=True, timeout=15,
    )
    output = proc.stdout + proc.stderr
    m = re.search(r"Python (\d+\.\d+)", output)
    return m.group(1) if m else None


def repo_test_hint(repo: str, python_version: str | None = None) -> str:
    """Hint for the generation prompt about the test environment."""
    if repo in _REPO_PREAMBLES:
        py_note = ""
        if python_version:
            py_note = (
                f"CRITICAL: The testbed runs Python {python_version}. "
                f"Only use syntax and stdlib features available in Python {python_version}. "
                "Avoid f-strings, walrus operator (:=), positional-only parameters, "
                "and any other syntax added after that version. "
            )
        return (
            py_note +
            "Django is already configured (DJANGO_SETTINGS_MODULE=test_sqlite, "
            "django.setup() already called). "
            "Do NOT call settings.configure() or django.setup() in your test. "
            "Do NOT use django.test.TestCase — it requires a migrated database. "
            "Write plain pytest functions instead. "
            "Validators return None on success and raise ValidationError on failure — "
            "use pytest.raises(ValidationError) to test them, not return-value assertions. "
            "Only import names you can see in the provided source files."
        )
    return ""


def fetch_relevant_code(
    instance: "object",
    max_chars_per_file: int = 3000,
) -> dict[str, str]:
    """Return {filepath: content} for source files changed by the gold patch.

    Reads files from the instance's Docker image at base_commit (the buggy
    version). Test files are excluded — showing those would hint at the
    answer. Returns an empty dict if the image isn't available.
    """
    instance_id: str = instance.instance_id  # type: ignore[attr-defined]
    gold_patch: str = instance.gold_patch    # type: ignore[attr-defined]
    image_name = f"sweb.eval.x86_64.{instance_id}:latest"

    # Extract file paths from diff headers, skip test files
    paths = [
        p for p in re.findall(r"^diff --git a/(.+) b/", gold_patch, re.MULTILINE)
        if not re.search(r"(^|/)tests?/|test_|_test\.py", p)
    ]

    result: dict[str, str] = {}
    for path in paths:
        proc = subprocess.run(
            ["docker", "run", "--rm", image_name, "cat", f"/testbed/{path}"],
            capture_output=True, text=True, timeout=15,
        )
        if proc.returncode != 0:
            continue
        content = proc.stdout
        if len(content) > max_chars_per_file:
            content = content[:max_chars_per_file] + "\n# ...[truncated]"
        result[path] = content
    return result


def make_test_patch(test_path: str, test_code: str) -> str:
    body = test_code if test_code.endswith("\n") else test_code + "\n"
    lines = body.split("\n")
    if lines and lines[-1] == "":
        lines = lines[:-1]
    hunk = "".join(f"+{ln}\n" for ln in lines)
    return (
        f"diff --git a/{test_path} b/{test_path}\n"
        f"new file mode 100644\n"
        f"--- /dev/null\n"
        f"+++ b/{test_path}\n"
        f"@@ -0,0 +1,{len(lines)} @@\n"
        f"{hunk}"
    )


@dataclass
class PredictionRecord:
    instance_id: str
    model_name_or_path: str
    model_patch: str

    def to_json(self) -> dict:
        return {
            "instance_id": self.instance_id,
            "model_name_or_path": self.model_name_or_path,
            "model_patch": self.model_patch,
        }


def write_predictions(
    records: list[PredictionRecord], path: Path
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for rec in records:
            f.write(json.dumps(rec.to_json()) + "\n")
    return path


@dataclass
class HarnessResult:
    instance_id: str
    resolved: bool
    fail_to_pass_passed: list[str]
    fail_to_pass_failed: list[str]
    pass_to_pass_passed: list[str]
    pass_to_pass_failed: list[str]
    applied: bool
    error: str | None = None


def run_harness(
    predictions_path: Path,
    dataset_split: str,
    run_id: str,
    instance_ids: list[str],
    *,
    max_workers: int = 1,
    build_locally: bool = True,
    cache_level: str = "instance",
) -> int:
    # cache_level="instance" keeps the eval Docker image after the run so
    # run_test_in_container can reuse it. Default "env" would remove it.

    dataset_name = {
        "lite": "princeton-nlp/SWE-bench_Lite",
        "verified": "princeton-nlp/SWE-bench_Verified",
    }[dataset_split]

    cmd = [
        sys.executable,
        "-m",
        "swebench.harness.run_evaluation",
        "--dataset_name",
        dataset_name,
        "--predictions_path",
        str(predictions_path),
        "--run_id",
        run_id,
        "--max_workers",
        str(max_workers),
        "--cache_level",
        cache_level,
        "--instance_ids",
        *instance_ids,
    ]
    if build_locally:
        cmd += ["--namespace", ""]

    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout[-4000:] + "\n" + proc.stderr[-4000:])
    _stash_summary_reports(run_id)
    return proc.returncode


HARNESS_SUMMARY_DIR = Path("logs") / "harness_reports"


def _stash_summary_reports(run_id: str) -> None:
    """Move the harness's `<model>.<run_id>.json` summary out of the cwd.

    The harness always writes this run-level summary to the current directory
    (its `--report_dir` flag is accepted but ignored). Nothing reads it — the
    verdict comes from the per-instance report under `report_dir` — so it is
    kept only for reference, under logs/.
    """
    for path in Path(".").glob(f"*.{run_id}.json"):
        HARNESS_SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
        path.replace(HARNESS_SUMMARY_DIR / path.name)


def report_dir(run_id: str, model_name: str, instance_id: str) -> Path:
    """Where the SWE-bench harness writes one instance's evaluation artifacts.

    The harness replaces "/" in the model name with "__" when it names this
    directory. Reading the raw name instead silently finds nothing for any model
    id containing a slash — every OpenRouter id does — and a missing report reads
    as "not resolved".
    """
    return (Path("logs") / "run_evaluation" / run_id
            / model_name.replace("/", "__") / instance_id)


def read_report(
    run_id: str,
    model_name: str,
    instance: "object",
) -> HarnessResult:
    instance_id: str = instance.instance_id  # type: ignore[attr-defined]
    report_path = report_dir(run_id, model_name, instance_id) / "report.json"

    if not report_path.exists():
        return HarnessResult(
            instance_id=instance_id,
            resolved=False,
            fail_to_pass_passed=[],
            fail_to_pass_failed=[],
            pass_to_pass_passed=[],
            pass_to_pass_failed=[],
            applied=False,
            error=f"report not found: {report_path}",
        )

    data = json.loads(report_path.read_text())
    inst = data[instance_id]
    ftp = inst.get("tests_status", {}).get("FAIL_TO_PASS", {})
    ptp = inst.get("tests_status", {}).get("PASS_TO_PASS", {})

    return HarnessResult(
        instance_id=instance_id,
        resolved=inst.get("resolved", False),
        fail_to_pass_passed=ftp.get("success", []),
        fail_to_pass_failed=ftp.get("failure", []),
        pass_to_pass_passed=ptp.get("success", []),
        pass_to_pass_failed=ptp.get("failure", []),
        applied=inst.get("patch_successfully_applied", False),
    )


def run_snippet_in_container(
    instance: "object",
    snippet: str,
    timeout: int = 60,
) -> str | None:
    """Run an arbitrary Python snippet inside the instance's Docker image.

    Returns stdout (up to 2000 chars) or None if the container errors out.
    Used in Phase 1 of two-phase generation to observe the buggy output.
    """
    instance_id: str = instance.instance_id  # type: ignore[attr-defined]
    image_name = f"sweb.eval.x86_64.{instance_id}:latest"

    repo: str = getattr(instance, "repo", "")
    preamble = _REPO_PREAMBLES.get(repo, "")

    with tempfile.TemporaryDirectory(dir=str(Path.home())) as tmpdir:
        tmp = Path(tmpdir)
        (tmp / "probe.py").write_text(preamble + snippet)

        script = "\n".join([
            "source /opt/miniconda3/bin/activate",
            "conda activate testbed",
            "cd /testbed",
            "python /patches/probe.py",
        ])

        cmd = [
            "docker", "run", "--rm",
            "-v", f"{tmpdir}:/patches:ro",
            image_name,
            "bash", "-c", script,
        ]
        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True, timeout=timeout
            )
        except subprocess.TimeoutExpired:
            return None

        output = proc.stdout.strip()
        if output:
            return output[:2000]
        # Non-zero exit with no stdout means the probe crashed — return the
        # traceback so Phase 2 knows the bug manifests as an exception.
        if proc.returncode != 0:
            err = (proc.stdout + proc.stderr).strip()
            return err[:2000] if err else None
        return None


@dataclass
class ContainerResult:
    """Outcome of running a test inside the Docker container.

    Behaves like a plain string (via __str__ / __eq__) so existing code
    that compares against "pass"/"fail"/etc. needs no changes.
    """
    outcome: str               # "pass" | "fail" | "error" | "absent"
    failure_reason: str | None = None  # "assertion" | "runtime" | None
    output: str | None = None  # tail of captured pytest output (for repair/debug)

    def __str__(self) -> str:
        return self.outcome

    def __eq__(self, other: object) -> bool:
        if isinstance(other, str):
            return self.outcome == other
        if isinstance(other, ContainerResult):
            return self.outcome == other.outcome
        return NotImplemented

    def __hash__(self) -> int:
        return hash(self.outcome)


def _parse_failure_reason(output: str) -> str | None:
    """Inspect pytest output to distinguish how a 'fail' outcome occurred.

    Returns:
      "assertion" — the test reached its assertion and it fired: an
                    AssertionError, a bare rewritten `assert`, or pytest's
                    own Failed (pytest.fail / DID NOT RAISE).
      "runtime"   — a non-assertion exception was raised before or instead
                    of an assertion (import error, Django mis-config,
                    missing fixture file, wrong API, etc.).
      None        — output is empty or carries no recognisable failure lines.

    Assertion evidence WINS. The question the gate actually asks is "did the
    test reach its assertion?", and if an assertion fired then it did — whatever
    else the output mentions. Scanning for a crash name first gets this backwards
    and rejects valid BRTs: astropy-12907's L4 test failed on `Expected:` inside
    its own assertion message being read as an exception class.

    Warnings are ignored entirely — a `DeprecationWarning` on an `E ` line is
    never why a test failed.
    """
    if not output:
        return None
    bodies = [m.group(1).strip() for m in _E_LINE_RE.finditer(output)]
    if not bodies:
        return None
    if any(_ASSERT_SIGNAL_RE.match(b) for b in bodies):
        return "assertion"
    for body in bodies:
        m = _EXC_NAME_RE.match(body)
        if m and _looks_like_exception(m.group(1)):
            return "runtime"
    return None


def _parse_pytest_outcome(returncode: int, output: str) -> str:
    # git apply failure — patch couldn't be applied to the tree
    if "error: patch failed" in output or "patch does not apply" in output:
        return "absent"
    # pytest exit codes: 0=pass, 1=fail, 5=no tests collected
    if returncode == 0:
        return "pass"
    if returncode == 1:
        return "fail"
    if returncode == 5:
        return "absent"
    # exit 2 (interrupted), 3 (internal error), 4 (bad CLI), or
    # docker/git errors all land here
    return "error"


def run_test_in_container(
    instance: "object",
    test_patch: str,
    *,
    gold_patch: str | None = None,
    timeout: int = 120,
    verbose: bool = False,
    test_path: str = BRT_TEST_PATH,
) -> ContainerResult:
    """Apply test_patch inside the instance's Docker image and run pytest.

    Runs twice from the pipeline:
      - gold_patch=None  → buggy tree  → expected "fail"
      - gold_patch=<str> → fixed tree  → expected "pass"

    Returns a ContainerResult whose .outcome is one of:
      "pass", "fail", "error", "absent"
    and whose .failure_reason is "assertion", "runtime", or None.

    The Docker image must already exist (built by run_harness / gold
    validation). Colima users: ensure DOCKER_HOST is set in the shell
    before calling this, or the subprocess docker call will fail.
    """
    instance_id: str = instance.instance_id  # type: ignore[attr-defined]
    image_name = f"sweb.eval.x86_64.{instance_id}:latest"

    # Colima (macOS Docker) only mounts ~/ into the VM by default.
    # /tmp on macOS is separate from the VM's /tmp, so always use home dir.
    with tempfile.TemporaryDirectory(dir=str(Path.home())) as tmpdir:
        tmp = Path(tmpdir)
        (tmp / "test.patch").write_text(test_patch)

        script_lines = [
            "set -e",
            "source /opt/miniconda3/bin/activate",
            "conda activate testbed",
            "cd /testbed",
            # Some testbed envs (e.g. Django) ship without pytest.
            # Install it silently; pip's local cache makes this ~instant on repeat runs.
            "python -m pytest --version > /dev/null 2>&1 || pip install pytest -q",
        ]
        if gold_patch is not None:
            (tmp / "gold.patch").write_text(gold_patch)
            script_lines.append("git apply /patches/gold.patch")
        script_lines.append("git apply /patches/test.patch")
        script_lines.append(f"python -m pytest {test_path} -x --tb=short -q")
        script = "\n".join(script_lines)

        cmd = [
            "docker", "run", "--rm",
            "-v", f"{tmpdir}:/patches:ro",
            image_name,
            "bash", "-c", script,
        ]

        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True, timeout=timeout
            )
        except subprocess.TimeoutExpired:
            return ContainerResult("error", output="timeout")

        if "Unable to find image" in proc.stderr:
            raise RuntimeError(
                f"Docker image {image_name!r} not found. "
                "Run scripts/01_validate_gold.py first to build it."
            )

        full_output = proc.stdout + proc.stderr
        outcome = _parse_pytest_outcome(proc.returncode, full_output)
        failure_reason = _parse_failure_reason(full_output) if outcome == "fail" else None

        if verbose or outcome == "error":
            label = "FIXED" if gold_patch is not None else "BUGGY"
            sys.stderr.write(
                f"\n--- container output [{label}] (rc={proc.returncode}) ---\n"
                + full_output[-3000:]
                + "\n---\n"
            )
        return ContainerResult(outcome, failure_reason, full_output[-3000:])
