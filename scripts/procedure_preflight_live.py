"""Run the procedure agent's preflight live on instances (Docker, no model call).

Checks the mechanics in real images: official runner vs fallback, the trivial
probes, issue-code runs, location candidates. Free: no tokens. Gold-free: the
preflight never sees the gold patch or the hidden tests.

Writes runs/preflight/<instance_id>.json and prints one summary line per instance.

    python3 scripts/procedure_preflight_live.py --ids django__django-11564 sympy__sympy-14396
    python3 scripts/procedure_preflight_live.py --round-file studies/issue-info-tests/data/dev/round1.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tdbf.agent.preflight import run_preflight  # noqa: E402
from tdbf.agent.session import ContainerSession  # noqa: E402
from tdbf.data.instances import InstanceStore  # noqa: E402

OUT = ROOT / "runs" / "preflight"


def summarise(pf) -> str:
    e = pf.env
    ran = [s for s in pf.snippets if s.ran]
    exc = [s.exception for s in ran if s.exception]
    top_code = pf.candidates.code[0].path if pf.candidates.code else "-"
    top_test = pf.candidates.tests[0].path if pf.candidates.tests else "-"
    runner = (e.command or {}).get("source", "-")
    return (f"{pf.instance_id}: env={'INVALID ' + str(e.reason) if e.invalid else 'ok'} "
            f"runner={runner} py={e.python_version} blocks={len(pf.snippets)} ran={len(ran)} "
            f"exc={exc} tracebacks={len(pf.tracebacks)} code1={top_code} test1={top_test} "
            f"secs={pf.seconds} dirty={pf.tree_dirty_after}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", nargs="*", default=[])
    ap.add_argument("--round-file", type=Path)
    args = ap.parse_args()
    ids = list(args.ids)
    if args.round_file:
        ids += json.loads(args.round_file.read_text())["instance_ids"]
    store = InstanceStore("lite")
    OUT.mkdir(parents=True, exist_ok=True)
    bad = 0
    for iid in ids:
        inst = store.get(iid)
        t0 = time.monotonic()
        try:
            with ContainerSession(inst) as session:
                pf = run_preflight(session, inst)
        except Exception as exc:  # noqa: BLE001 - report and continue
            print(f"{iid}: CRASH {type(exc).__name__}: {exc}", flush=True)
            bad += 1
            continue
        rec = pf.to_json()
        rec["wall_seconds"] = round(time.monotonic() - t0, 1)
        (OUT / f"{iid}.json").write_text(json.dumps(rec, indent=2))
        print(summarise(pf) + f" wall={rec['wall_seconds']}s", flush=True)
        bad += int(pf.env.invalid)
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
