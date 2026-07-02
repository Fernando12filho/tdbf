"""TDBF — Test-Driven Bug Fixing. Phase 1: pre-fix test generation."""

from __future__ import annotations

import os
from pathlib import Path

__version__ = "0.1.0"


def load_env(start: Path | None = None) -> None:
    """Load KEY=VALUE pairs from the nearest `.env` into os.environ.

    Zero-dependency (no python-dotenv). Walks up from `start` (or cwd) to the
    filesystem root looking for a `.env`, then sets any key that is NOT already
    present in the environment — real env vars always win, and an empty value in
    `.env` never clobbers a set variable. Missing/blank files are ignored.

    Called once on package import so scripts and the webapp pick up API keys
    without a manual `source .env`.
    """
    here = (start or Path.cwd()).resolve()
    for base in [here, *here.parents]:
        env_file = base / ".env"
        if not env_file.is_file():
            continue
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key, value = key.strip(), value.strip().strip("'\"")
            if key and value and key not in os.environ:
                os.environ[key] = value
        return  # first .env found wins


load_env()
