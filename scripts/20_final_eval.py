#!/usr/bin/env python3
"""Final evaluation driver (studies/final-eval/plan.md): TDBF's single agent and the
reactive blackboard against ReProAgent on SWT-bench Lite, one bug at a time.

Per bug:
  1. make sure the prebuilt SWE-bench image is local (pull, tag for ContainerSession)
  2. clean bugs: run the single agent (run group final-single) and the reactive
     blackboard (final-rbb) through scripts/scouts_brt.py; both resume by run group
  3. export every system's test as an SWT-bench prediction (runs/final/*.jsonl)
  4. score gold + single + reactive + ReProAgent in one pass on the same image
     (scripts/18_swt_score.py, in the SWT-bench checkout's own venv)
  5. read the four reports into studies/final-eval/results/verdicts.jsonl, refusing a
     report older than this bug's scoring step
  6. remove the image, unless it was here before or it is a development bug

The 50 development bugs (--set dev) reuse stored runs of the same configurations
and are only exported and scored. Resumable: a bug whose verdicts are all stored is
skipped. Stops on an out-of-credit / key-limit error, when the agents' cost reaches
--max-usd, or when free disk falls below --min-free-gb.

    python3 scripts/20_final_eval.py --set clean --limit 3        # pilot
    python3 scripts/20_final_eval.py --set dev                    # score the reused 50
    python3 scripts/20_final_eval.py --set clean --dry-run
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tdbf import metrics  # noqa: E402
from tdbf.harness.runner import brt_run_spec, make_test_patch  # noqa: E402

SAMPLE = ROOT / "studies" / "final-eval" / "data" / "sample_swt_lite_final.json"
STORE = ROOT / "studies" / "final-eval" / "results" / "verdicts.jsonl"
PRED_DIR = ROOT / "runs" / "final"
RESULTS = ROOT / "results.jsonl"
SWT_DIR = Path.home() / "swt-bench"
SWT_PY = SWT_DIR / ".venv" / "bin" / "python"
REPRO_SRC = (ROOT.parents[2] / "external" / "reproagent" / "repo" / "outputs_gpt_5_mini"
             / "repro_agent_preds_gpt_5_mini_lite.jsonl")
GOLD_RUN = "validate-lite-gold-1"
MINI = "openrouter:openai/gpt-5-mini@openai"
NANO = "openrouter:openai/gpt-5-nano@openai"

# system -> (run id, model name in its predictions, run groups its tests come from)
SYSTEMS = {
    "single": ("final-single-v1", "tdbf__single-gpt-5-mini",
               ["final-single", "v2-base", "proc-dev-baseline"]),
    "rbb": ("final-rbb-v1", "tdbf__rbb-gpt-5-mini", ["final-rbb", "rbb-fresh", "rbb-dev30"]),
    "reproagent": ("final-reproagent-v1", "ReProAgent", []),
    # studies/multi-test: the multi-test BRTs, scored with the same harness on the dev bugs
    "multi": ("mt-multi-v1", "tdbf__multi-gpt-5-mini", ["mt-brt"]),
}
OURS = ("single", "rbb", "multi")
DEFAULT_SYSTEMS = ("single", "rbb", "reproagent")
_PROC_REF = "agent/procedure:results.jsonl"  # proc-dev-baseline lives on that branch


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


SWT_PLACEMENT = _load("swt17", "17_swt_predictions.py").SWT_PLACEMENT


def sh(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    print("+", " ".join(str(c) for c in cmd), flush=True)
    return subprocess.run(cmd, **kw)


def registry_image(iid: str) -> str:
    return f"swebench/sweb.eval.x86_64.{iid.lower().replace('__', '_1776_')}:latest"


def local_image(iid: str) -> str:
    return f"sweb.eval.x86_64.{iid}:latest"


def image_exists(name: str) -> bool:
    return subprocess.run(["docker", "image", "inspect", name], capture_output=True).returncode == 0


def free_gb() -> float:
    return shutil.disk_usage(Path.home()).free / 1e9


# ── tests -> predictions ─────────────────────────────────────────────────────

def _records() -> list[dict]:
    recs = metrics.load_records(RESULTS)
    try:
        proc = subprocess.run(["git", "-C", str(ROOT), "show", _PROC_REF],
                              capture_output=True, text=True, check=True).stdout
        recs += [json.loads(l) for l in proc.splitlines() if '"proc-dev-baseline"' in l]
    except subprocess.CalledProcessError:
        print(f"[warn] could not read {_PROC_REF}; proc-dev-baseline tests unavailable")
    return recs


def tests_of(system: str, recs: list[dict]) -> dict[str, dict]:
    """instance -> {repo, test_code, group}: the latest record per instance, from
    the system's groups in priority order (final runs first)."""
    out: dict[str, dict] = {}
    for group in reversed(SYSTEMS[system][2]):  # later groups overwrite earlier ones
        for r in recs:
            if (r.get("flags") or {}).get("run_group") != group:
                continue
            att = metrics._best_attempt(r) if r.get("attempts") else {}
            out[r["instance_id"]] = {"repo": r["repo"], "test_code": att.get("test_code") or "",
                                     "group": group, "level": att.get("level")}
    return out


def prediction(repo: str, test_code: str, model_name: str, iid: str) -> dict:
    spec = brt_run_spec(repo, test_code)
    sub = SWT_PLACEMENT.get(repo)
    if sub:
        spec.files = {f"{sub}/{Path(p).name}": c for p, c in spec.files.items()}
    patch = "".join(make_test_patch(p, c) for p, c in spec.files.items()) if test_code.strip() else ""
    return {"instance_id": iid, "model_name_or_path": model_name, "model_patch": patch}


def write_predictions(ids: list[str], systems=DEFAULT_SYSTEMS) -> dict[str, Path]:
    """(Re)write each system's predictions file for `ids` from the stored tests."""
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    recs = _records()
    paths = {}
    for system in [s for s in systems if s in OURS]:
        _, model, _ = SYSTEMS[system]
        tests = tests_of(system, recs)
        path = PRED_DIR / f"{system}.jsonl"
        with path.open("w") as f:
            for iid in ids:
                t = tests.get(iid)
                if t and t["test_code"].strip():
                    f.write(json.dumps(prediction(t["repo"], t["test_code"], model, iid)) + "\n")
        paths[system] = path
    if "reproagent" not in systems:
        return paths
    frozen = json.loads(SAMPLE.read_text())["reproagent_predictions"]["sha256"]
    raw = REPRO_SRC.read_bytes()
    if hashlib.sha256(raw).hexdigest() != frozen:
        raise SystemExit(f"ReProAgent predictions differ from the frozen sha256: {REPRO_SRC}")
    path = PRED_DIR / "reproagent.jsonl"
    path.write_bytes(raw)
    paths["reproagent"] = path
    return paths


# ── verdicts ─────────────────────────────────────────────────────────────────

def stored() -> dict[tuple[str, str], dict]:
    out = {}
    if STORE.exists():
        for line in STORE.read_text().splitlines():
            if line.strip():
                v = json.loads(line)
                out[(v["instance_id"], v["system"])] = v
    return out


def _report(run_id: str, model: str, iid: str) -> Path:
    return SWT_DIR / "run_instance_swt_logs" / run_id / model / iid / "report.json"


def collect(iid: str, started: float, preds: dict[str, Path]) -> list[dict]:
    """Read the gold report and each system's report for `iid`; refuse stale ones."""
    out = []
    now = datetime.now(timezone.utc).isoformat()
    gold_path = _report(GOLD_RUN, "gold", iid)
    gold_ok = None
    if gold_path.exists():
        g = json.loads(gold_path.read_text()).get(iid, {})
        gold_ok = bool(g.get("resolved"))
    out.append({"instance_id": iid, "system": "gold", "S": gold_ok, "scored_at": now})
    for system, path in preds.items():
        rid, model, _ = SYSTEMS[system]
        pred = next((json.loads(l) for l in path.read_text().splitlines()
                     if l.strip() and json.loads(l)["instance_id"] == iid), None)
        v = {"instance_id": iid, "system": system, "run_id": rid, "scored_at": now,
             "gold_ok": gold_ok}
        if pred is None or not pred["model_patch"].strip():
            v.update(S=False, applied=False, note="no prediction (counts as a failure)")
            out.append(v)
            continue
        rp = _report(rid, model, iid)
        if not rp.exists() or rp.stat().st_mtime < started:
            v.update(S=None, note="no fresh report", stale=rp.exists())
            out.append(v)
            continue
        rep = json.loads(rp.read_text()).get(iid, {})
        added = lambda k: bool(set(rep.get("tests_pred", {}).get(k, []))  # noqa: E731
                               - set(rep.get("tests_base", {}).get(k, [])))
        applied = bool(rep.get("patch_successfully_applied"))
        v.update(S=bool(rep.get("resolved")), applied=applied,
                 f2p=applied and added("FAIL_TO_PASS"),
                 coverage_delta=rep.get("coverage_delta_pred"),
                 patch_sha256=hashlib.sha256(pred["model_patch"].encode()).hexdigest())
        out.append(v)
    return out


# ── one bug ──────────────────────────────────────────────────────────────────

class Stop(Exception):
    pass


def run_agents(iid: str, args) -> None:
    common = ["--bugs-file", str(SAMPLE.parent / "clean_ids.json"), "--ids", iid,
              "--model", MINI, "--max-usd", "1"] + (["--no-label"] if args.no_label else [])
    arms = [("base", "final-single", []),
            ("rbb", "final-rbb", ["--scout-turns", "50",
                                  "--role-model", f"template={NANO}",
                                  "--role-model", f"localizer={NANO}",
                                  "--role-model", f"reproducer={NANO}"])]
    for arm, group, extra in arms:
        res = sh([sys.executable, str(ROOT / "scripts" / "scouts_brt.py"), "--arm", arm,
                  "--run-group", group, *common, *extra], capture_output=True, text=True)
        log = PRED_DIR / "logs" / f"{iid}.{arm}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text(res.stdout + res.stderr)
        tail = (res.stdout + res.stderr)[-3000:]
        if "Error code: 402" in tail or "Error code: 403" in tail:
            raise Stop(f"{iid} {arm}: OpenRouter credits or key limit exhausted (see {log})")
        if "CRASH" in tail or res.returncode != 0:
            print(f"  [warn] {iid} {arm}: crashed or failed (see {log}); its test counts as missing")


def agents_cost() -> float:
    total = 0.0
    for r in metrics.load_records(RESULTS):
        if (r.get("flags") or {}).get("run_group") in ("final-single", "final-rbb"):
            total += float(r.get("cost_total_usd") or
                           sum((a.get("cost_usd") or 0) for a in r.get("attempts", [])))
    return total


def score(iid: str, preds: dict[str, Path]) -> float:
    # a fresh report for every system: remove this bug's old reports first
    for system in preds:
        rid, model, _ = SYSTEMS[system]
        old = _report(rid, model, iid).parent
        if old.exists():
            shutil.rmtree(old)
    started = time.time()
    names = list(preds)
    cmd = [str(SWT_PY), str(ROOT / "scripts" / "18_swt_score.py"),
           "--predictions", str(preds[names[0]]), "--run-id", SYSTEMS[names[0]][0]]
    for name in names[1:]:
        cmd += ["--also", f"{preds[name]}={SYSTEMS[name][0]}"]
    cmd += ["--ids", iid]
    res = sh(cmd, cwd=str(SWT_DIR), capture_output=True, text=True)
    log = PRED_DIR / "logs" / f"{iid}.score.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(res.stdout + res.stderr)
    return started


def do_bug(iid: str, args, dev: set[str]) -> None:
    print(f"\n{'=' * 70}\n{iid}  ({'development: reuse' if iid in dev else 'clean'})", flush=True)
    had_local = image_exists(local_image(iid))
    had_registry = image_exists(registry_image(iid))
    if not had_registry:
        if free_gb() < args.min_free_gb:
            raise Stop(f"free disk {free_gb():.0f} GB < {args.min_free_gb} GB")
        if not any(sh(["docker", "pull", "--platform", "linux/amd64", registry_image(iid)])
                   .returncode == 0 for _ in range(3)):
            print(f"  [skip] {iid}: pull failed")
            return
    if not had_local:
        sh(["docker", "tag", registry_image(iid), local_image(iid)], check=True)
    try:
        if iid not in dev:
            run_agents(iid, args)
        preds = write_predictions([iid], args.systems)
        started = score(iid, preds)
        verdicts = collect(iid, started, preds)
        with STORE.open("a") as f:
            for v in verdicts:
                f.write(json.dumps(v) + "\n")
        print("  verdicts: " + ", ".join(f"{v['system']}={v['S']}" for v in verdicts), flush=True)
    finally:
        if not args.keep_images and iid not in dev:
            if not had_local:
                sh(["docker", "rmi", local_image(iid)], capture_output=True)
            if not had_registry:
                sh(["docker", "rmi", registry_image(iid)], capture_output=True)


def done(iid: str, have: dict, systems) -> bool:
    need = ["gold", *systems]
    return all((iid, s) in have and have[(iid, s)].get("S") is not None for s in need)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", choices=["clean", "dev"], required=True)
    ap.add_argument("--ids", nargs="*", default=[])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-usd", type=float, default=30.0, help="agents' total cost cap")
    ap.add_argument("--min-free-gb", type=float, default=20.0)
    ap.add_argument("--keep-images", action="store_true")
    ap.add_argument("--no-label", action="store_true",
                    help="skip our offline L0-L4 label after the agents (plan amendment 1)")
    ap.add_argument("--systems", nargs="+", default=list(DEFAULT_SYSTEMS),
                    choices=list(SYSTEMS), help="systems to score (default: the evaluation's three)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    sample = json.loads(SAMPLE.read_text())
    dev = set(sample["development_ids"])
    (SAMPLE.parent / "clean_ids.json").write_text(json.dumps({"instance_ids": sample["clean_ids"]}))
    pool = sample["clean_ids"] if args.set == "clean" else sample["development_ids"]
    ids = [i for i in (args.ids or pool) if i in pool]
    have = stored()
    todo = [i for i in ids if not done(i, have, args.systems)]
    # images already on disk first
    todo.sort(key=lambda i: not image_exists(registry_image(i)))
    if args.limit:
        todo = todo[:args.limit]
    print(f"set={args.set} todo={len(todo)}/{len(ids)} agents' cost so far ${agents_cost():.2f} "
          f"cap ${args.max_usd} free disk {free_gb():.0f} GB")
    if args.dry_run:
        print("\n".join(todo))
        return 0
    if not SWT_PY.exists():
        raise SystemExit(f"SWT-bench venv not found: {SWT_PY}")
    STORE.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    for n, iid in enumerate(todo, 1):
        if args.set == "clean" and agents_cost() >= args.max_usd:
            print(f"cost cap reached (${agents_cost():.2f} >= ${args.max_usd}); stopping.")
            break
        try:
            do_bug(iid, args, dev)
        except Stop as e:
            print(f"STOP: {e}")
            return 2
        except Exception as e:  # noqa: BLE001 - report, next bug
            print(f"{iid}: CRASH {type(e).__name__}: {e}", flush=True)
        print(f"[{n}/{len(todo)}] elapsed {(time.monotonic() - t0) / 3600:.1f} h, "
              f"agents' cost ${agents_cost():.2f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
