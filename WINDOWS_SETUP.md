# Windows Setup Guide

Full setup for running TDBF on Windows with Docker Desktop.

## Prerequisites

### 1. Python 3.10+
Download and install from [python.org](https://www.python.org/downloads/windows/).
During installation, check **"Add Python to PATH"**.

Verify:
```
python --version
```

### 2. Docker Desktop with WSL2
Download from [docker.com/products/docker-desktop](https://www.docker.com/products/docker-desktop/).

Requirements:
- Enable WSL2 backend during installation (recommended, required for SWE-bench images)
- Allocate at least **8 GB RAM** to Docker: Docker Desktop → Settings → Resources → Memory → 8 GB
- Allocate at least **4 CPUs**

> SWE-bench images are built for `linux/x86_64`. On Windows/WSL2 this runs natively — no emulation overhead like on Apple Silicon.

### 3. Git
Download from [git-scm.com](https://git-scm.com/download/win) if not already installed.

---

## Install

Clone and install the package in editable mode:
```
git clone https://github.com/<your-org>/tdbf.git
cd tdbf
pip install -e .
```

Install runtime dependencies (needed for live runs, not the smoke test):
```
pip install swebench datasets anthropic openai
```

---

## API Keys

Set environment variables via PowerShell (session-only):
```powershell
$env:ANTHROPIC_API_KEY = "sk-ant-..."
$env:OPENAI_API_KEY    = "sk-..."
```

Or permanently via System Properties → Environment Variables → User variables.

---

## Docker socket

No extra configuration needed. Docker Desktop on Windows automatically exposes the socket at the default path that the `docker` Python SDK expects. Unlike macOS/Colima, you do **not** need to set `DOCKER_HOST`.

---

## Verify the setup

**Step 1 — offline smoke test** (no API keys, no Docker):
```
python scripts/00_smoke_test.py
```
Expected output: `ALL SMOKE CHECKS PASSED`

**Step 2 — gold patch validation** (requires Docker running):
```
python scripts/01_validate_gold.py --split lite
```
This builds a per-instance Docker image (~5–10 min on first run, cached after) and confirms the harness correctly scores the gold patch as resolved. Expected: `Harness exit code: 0` and the instance appears in `resolved_ids` in the summary JSON.

---

## Notes

- `runs/` and `logs/` are gitignored — harness output stays local.
- The harness summary JSON (e.g. `GOLD.gold_validation.json`) is written to the repo root and is also gitignored.
- Use `python` on Windows (not `python3`).
