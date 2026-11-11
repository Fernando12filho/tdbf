"""Result persistence: append-only jsonl log + optional webapp POST.

The pipeline always writes to results.jsonl regardless of whether the
webapp is running. The webapp POST is best-effort — a failed connection
is silently swallowed so the pipeline never blocks on the UI.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

RESULTS_FILE = Path("results.jsonl")
_WEBAPP_URL = os.environ.get("TDBF_WEBAPP_URL", "http://localhost:5000")

# Set by use_run_group() for scripts that build their flags without one.
_DEFAULT_RUN_GROUP: str | None = None


def use_run_group(run_group: str | None, model: str) -> None:
    """Check `run_group` against the study registry BEFORE any model call, and
    tag every record this process saves with it (when its flags carry none).

    Scripts call this right after building their client, so a run that would be
    refused at save time is refused before it spends anything."""
    global _DEFAULT_RUN_GROUP
    from .studies import check_write
    check_write(run_group, model)
    _DEFAULT_RUN_GROUP = run_group


_PROVENANCE: dict | None = None


def code_provenance() -> dict:
    """The code a run was made with: the git commit of the checkout this package
    is imported from, and whether that checkout had uncommitted changes.

    Read once per process. Records written before 2026-11-11 have no `code` key;
    for them the commit can only be inferred from dates. A run made with
    uncommitted changes (`dirty: true`) cannot be reproduced from the commit alone.
    """
    global _PROVENANCE
    if _PROVENANCE is None:
        import subprocess
        here = Path(__file__).resolve().parent

        def git(*args: str) -> str | None:
            try:
                out = subprocess.run(["git", "-C", str(here), *args], capture_output=True,
                                     text=True, timeout=10)
                return out.stdout.strip() if out.returncode == 0 else None
            except (OSError, subprocess.SubprocessError):
                return None

        status = git("status", "--porcelain", "--untracked-files=no")
        _PROVENANCE = {"commit": git("rev-parse", "HEAD"),
                       "dirty": None if status is None else bool(status)}
    return _PROVENANCE


def save_run(record: dict) -> None:
    """Write record to results.jsonl and try to POST it to the webapp.

    record should contain at minimum:
        instance_id, repo, model, split, flags (dict),
        best_level (int), best_attempt (int), attempts (list of dicts)

    Refused (StudyError, nothing written) unless its run group is registered in
    `tdbf.studies`, the group's study is open, and the record's model is the
    study's: see studies.check_write.
    """
    from .studies import check_write
    flags = dict(record.get("flags") or {})
    if not flags.get("run_group") and _DEFAULT_RUN_GROUP:
        flags["run_group"] = _DEFAULT_RUN_GROUP
    check_write(flags.get("run_group"), record.get("model"))
    record = {**record, "flags": flags, "timestamp": datetime.now(timezone.utc).isoformat(),
              "code": code_provenance()}

    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS_FILE.open("a") as f:
        f.write(json.dumps(record) + "\n")

    _try_post(record)


def load_best_brt(instance_id: str, *, run_groups: tuple[str, ...],
                  models: tuple[str, ...] | None = None, min_level: int = 4) -> dict | None:
    """The best logged Bug Reproduction Test for an instance, from the given run
    groups only (and, if given, those models only).

    Used by the Phase B fix agent (script 05) to obtain the BRT it must make pass.
    Until 2026-11-03 this searched every record of every model and run, so a fix
    run could silently get a test another model wrote; the caller now names where
    the test may come from (by default its own study, see script 05).

    Highest level wins, then the most recent record. Returns {"test_code",
    "level", "model", "run_group", "timestamp"} or None.
    """
    if not RESULTS_FILE.exists():
        return None
    best: dict | None = None
    with RESULTS_FILE.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("instance_id") != instance_id:
                continue
            group = (rec.get("flags") or {}).get("run_group")
            if group not in run_groups or (models and rec.get("model") not in models):
                continue
            for att in rec.get("attempts", []):
                level = att.get("level") or 0
                code = att.get("test_code")
                if not code or level < min_level:
                    continue
                cand = {"test_code": code, "level": level, "model": rec.get("model", ""),
                        "run_group": group, "timestamp": rec.get("timestamp", "")}
                if best is None or (level, cand["timestamp"]) >= (best["level"], best["timestamp"]):
                    best = cand
    return best


def _try_post(record: dict) -> None:
    try:
        import urllib.request

        data = json.dumps(record).encode()
        req = urllib.request.Request(
            f"{_WEBAPP_URL}/api/runs",
            data=data,
            headers={"Content-Type": "application/json"},
        )
        urllib.request.urlopen(req, timeout=2)
    except Exception:
        pass
