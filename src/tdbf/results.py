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
