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


def save_run(record: dict) -> None:
    """Write record to results.jsonl and try to POST it to the webapp.

    record should contain at minimum:
        instance_id, repo, model, split, flags (dict),
        best_level (int), best_attempt (int), attempts (list of dicts)
    """
    record = {**record, "timestamp": datetime.now(timezone.utc).isoformat()}

    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS_FILE.open("a") as f:
        f.write(json.dumps(record) + "\n")

    _try_post(record)


def load_best_brt(instance_id: str, *, min_level: int = 4) -> dict | None:
    """Find the best logged Bug Reproduction Test for an instance.

    Scans results.jsonl for runs of `instance_id` and returns the test_code of
    the highest-scoring attempt that meets `min_level` (default L4 = validated).
    Used by the Phase B fix agent to obtain the BRT it must make pass.

    Returns {"test_code", "level", "model"} or None if no qualifying BRT exists.
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
            for att in rec.get("attempts", []):
                level = att.get("level", 0)
                code = att.get("test_code")
                if not code or level < min_level:
                    continue
                if best is None or level > best["level"]:
                    best = {"test_code": code, "level": level, "model": rec.get("model", "")}
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
