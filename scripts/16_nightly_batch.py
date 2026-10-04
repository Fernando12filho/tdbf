"""Nightly batch: pull -> gold check -> paired unit -> delete, one instance at a time.

For each instance of a frozen sample, in order:

  1. Skip it if its paired unit is already recorded, or its environment record
     already marks it not-valid. Stop for the night if the day's quota cannot
     cover a full pair (--min-remaining) or Docker is short of disk.
  2. Image: use the local `sweb.eval.x86_64.<id>:latest` if present; otherwise
     pull the official prebuilt image (linux/amd64) and tag it under both names
     -- the registry name the harness looks for (so it builds nothing) and the
     short name the agent containers use. The digest is recorded.
  3. Gold check + BRT mechanics (script 13's check_instance, no model calls). A
     failure marks the instance environment-invalid BEFORE any arm runs, which
     is the frozen sample's rule.
  4. The paired unit: script 11's run_sweep on this one instance, both arms in
     the seeded order. Records are written only when both arms finish.
  5. Delete the image if this batch pulled it, then prune dangling layers.

The environment record (default: the sample file with "_frozen" -> "_env") is
rewritten after every instance, in the same format as sample_lite_env.json.
Only one batch runs at a time (lock file). Keep the machine awake:

    caffeinate -i python3 scripts/16_nightly_batch.py \\
        --sample sample_lite_ext_frozen.json --run-group rq1-ext \\
        --model openrouter:nvidia/nemotron-3-super-120b-a12b:free

    # one-instance trial, or a plan without Docker or model calls
    python3 scripts/16_nightly_batch.py ... --ids django__django-11099 --max-instances 1
    python3 scripts/16_nightly_batch.py ... --dry-run

Exit status: 0 nothing left or limit reached, 3 quota, 4 repeated failures,
5 disk, 6 stop time reached. Re-running the same command resumes.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tdbf.harness.runner import registry_image  # noqa: E402

EXIT_DONE, EXIT_QUOTA, EXIT_FAILURES, EXIT_DISK, EXIT_TIME = 0, 3, 4, 5, 6
MIN_REMAINING = 140   # calls; the most any RQ1 pair has used is 132 (median 75)
MIN_FREE_GB = 25.0    # largest prebuilt image is ~11 GB (matplotlib), plus headroom


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / filename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def next_utc(hhmm: str, now: datetime | None = None) -> datetime:
    """The next time the UTC clock reads HH:MM (today if still ahead, else tomorrow)."""
    hh, mm = (int(x) for x in hhmm.split(":"))
    now = now or datetime.now(timezone.utc)
    at = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
    return at if at > now else at + timedelta(days=1)


def short_image(instance_id: str) -> str:
    return f"sweb.eval.x86_64.{instance_id}:latest"


# ── Docker ────────────────────────────────────────────────────────────────────

class Docker:
    """The few Docker operations the batch needs (swapped for a fake in the smoke)."""

    def __init__(self) -> None:
        sock = Path.home() / ".colima" / "default" / "docker.sock"
        if not os.environ.get("DOCKER_HOST") and sock.exists():
            os.environ["DOCKER_HOST"] = f"unix://{sock}"

    @staticmethod
    def _run(*args: str, timeout: float = 120) -> subprocess.CompletedProcess:
        return subprocess.run(["docker", *args], capture_output=True, text=True,
                              timeout=timeout)

    def has(self, image: str) -> bool:
        return self._run("image", "inspect", image).returncode == 0

    def free_gb(self) -> float | None:
        """Free space on Docker's data disk (the Colima VM disk on a Mac), read
        from inside a tiny container. None if it cannot be measured."""
        try:
            p = self._run("run", "--rm", "alpine:3", "df", "-Pk", "/", timeout=300)
        except subprocess.TimeoutExpired:
            return None
        lines = p.stdout.strip().splitlines()
        if p.returncode != 0 or len(lines) < 2:
            return None
        return int(lines[-1].split()[3]) / 1024 / 1024

    def pull(self, image: str) -> tuple[bool, str]:
        try:
            p = self._run("pull", "--platform", "linux/amd64", image, timeout=3600)
        except subprocess.TimeoutExpired:
            return False, "docker pull timed out after 3600s"
        return p.returncode == 0, (p.stderr or p.stdout).strip()[-500:]

    def tag(self, src: str, dst: str) -> bool:
        return self._run("tag", src, dst).returncode == 0

    def digest(self, image: str) -> str | None:
        p = self._run("image", "inspect", "--format", "{{index .RepoDigests 0}}", image)
        return (p.stdout.strip() or None) if p.returncode == 0 else None

    def remove(self, *images: str) -> None:
        self._run("rmi", *images, timeout=600)
        self._run("image", "prune", "-f", timeout=600)


# ── environment record ────────────────────────────────────────────────────────

class EnvRecord:
    """Per-instance environment state, same shape as sample_lite_env.json."""

    def __init__(self, path: Path, sample_name: str) -> None:
        self.path = path
        data = json.loads(path.read_text()) if path.exists() else {}
        self.sample = data.get("sample", sample_name)
        self.instances: dict[str, dict] = data.get("instances", {})

    def state(self, iid: str) -> str | None:
        return (self.instances.get(iid) or {}).get("state")

    def set(self, iid: str, **fields) -> None:
        self.instances.setdefault(iid, {}).update(fields)
        counts: dict[str, int] = {}
        for row in self.instances.values():
            counts[row.get("state", "?")] = counts.get(row.get("state", "?"), 0) + 1
        self.path.write_text(json.dumps({
            "sample": self.sample, "written_utc": utcnow(), "counts": counts,
            "instances": self.instances}, indent=2) + "\n")


# ── the batch ─────────────────────────────────────────────────────────────────

@dataclass
class Batch:
    status: int = EXIT_DONE
    stop_reason: str | None = None
    done: list[str] = field(default_factory=list)
    excluded: list[tuple[str, str]] = field(default_factory=list)
    failed: list[tuple[str, str]] = field(default_factory=list)
    skipped: int = 0


def ensure_image(docker: Docker, iid: str) -> tuple[bool, bool, dict]:
    """(available, pulled_by_us, provenance). Uses a local image when present."""
    short, reg = short_image(iid), registry_image(iid)
    if docker.has(short):
        if docker.has(reg):
            return True, False, {"image": f"{reg} @ {docker.digest(reg)}"}
        return True, False, {"image": "local image (already present)"}
    print(f"  pulling {reg} ...", flush=True)
    ok, msg = docker.pull(reg)
    if not ok:
        return False, False, {"reason": f"prebuilt image unavailable: {msg}"}
    if not docker.tag(reg, short):
        docker.remove(reg)
        return False, False, {"reason": f"could not tag {reg} as {short}"}
    return True, True, {"image": f"{reg} @ {docker.digest(reg)}", "pulled_utc": utcnow()}


def run_batch(ids: list[str], *, store, cfg, env: EnvRecord, docker: Docker,
              m11, check_instance, max_instances: int | None, stop_at: datetime | None,
              min_remaining: int, min_free_gb: float, max_consecutive_failures: int = 3,
              keep_images: bool = False, results_path: Path | None = None) -> Batch:
    b = Batch()
    usage = m11.Usage(cfg.out_dir / cfg.run_group / "usage.json", cfg.daily_budget)
    model_name = m11.build_client(cfg.model, num_ctx=cfg.num_ctx).name
    done = m11.completed_units(results_path or m11.RESULTS_FILE, model_name,
                               cfg.run_group, cfg.arms)
    consecutive = 0
    for iid in ids:
        if iid in done:
            b.skipped += 1
            continue
        state = env.state(iid)
        if state not in (None, "valid", "pending"):
            b.skipped += 1          # decided earlier: never re-run an excluded instance
            continue
        if max_instances is not None and len(b.done) + len(b.failed) >= max_instances:
            b.stop_reason = f"--max-instances {max_instances} reached"
            break
        if stop_at is not None and datetime.now(timezone.utc) >= stop_at:
            b.status, b.stop_reason = EXIT_TIME, f"stop time {stop_at.isoformat()} reached"
            break
        if usage.remaining() < min_remaining:
            b.status, b.stop_reason = EXIT_QUOTA, (
                f"daily budget: {usage.used_today()}/{cfg.daily_budget} used, "
                f"a pair can need up to {min_remaining}")
            break
        free = docker.free_gb()
        if free is not None and free < min_free_gb:
            b.status, b.stop_reason = EXIT_DISK, f"Docker disk: {free:.1f} GB free < {min_free_gb}"
            break

        inst = store.get(iid)
        print(f"\n{'=' * 64}\n[{utcnow()}] {iid}  (free disk "
              f"{'?' if free is None else f'{free:.0f} GB'}, calls today "
              f"{usage.used_today()}/{cfg.daily_budget})", flush=True)
        ok, pulled, prov = ensure_image(docker, iid)
        if not ok:
            env.set(iid, state="image_unavailable", reason=prov["reason"],
                    gold_resolved=None, checked_utc=utcnow())
            b.excluded.append((iid, "image_unavailable"))
            print(f"  [excluded] {prov['reason']}")
            continue
        # A valid instance keeps its state (it is only being re-run after a failed
        # unit); anything else is pending until the gold check decides.
        env.set(iid, **({} if state == "valid" else {"state": "pending"}), **prov)
        try:
            if state != "valid":    # gold + mechanics check, no model calls
                st = check_instance(inst, cfg.split, cfg.out_dir / cfg.run_group / "gold")
                if not st["valid"]:
                    reason = st.get("gold_error") or st.get("brt_error") or (
                        "gold patch does not resolve" if st.get("gold_resolved") is False
                        else "environment check failed")
                    env.set(iid, state="environment_invalid", reason=reason,
                            gold_resolved=st.get("gold_resolved"),
                            checked_utc=st["checked_utc"])
                    b.excluded.append((iid, "environment_invalid"))
                    print(f"  [excluded] environment invalid: {reason}")
                    continue
                env.set(iid, state="valid", reason=None, gold_resolved=True,
                        checked_utc=st["checked_utc"])

            s = m11.run_sweep(cfg, [inst])          # the paired unit, unchanged
            if s.status == m11.EXIT_QUOTA:
                b.status, b.stop_reason = EXIT_QUOTA, s.stop_reason
                break
            if s.done:
                b.done.append(iid)
                consecutive = 0
            elif s.failed:
                b.failed.append(s.failed[0])
                consecutive += 1
                if consecutive >= max_consecutive_failures:
                    b.status, b.stop_reason = EXIT_FAILURES, (
                        f"{consecutive} consecutive unit failures — last: {s.failed[0][1]}")
                    break
        finally:
            # Sequential and lock-protected, so no harness run is starting now:
            # deleting here cannot pull an image out from under one.
            if pulled and not keep_images:
                print(f"  removing {iid} image", flush=True)
                docker.remove(registry_image(iid), short_image(iid))
    return b


def _lock(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise SystemExit(f"another batch holds {path} (pid {path.read_text().strip()}); "
                         "if none is running, delete the lock file and retry")
    os.write(fd, str(os.getpid()).encode())
    os.close(fd)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--sample", required=True, help="frozen sample file ({'instances': [...]})")
    ap.add_argument("--env-record", help="default: the sample path with _frozen -> _env")
    ap.add_argument("--run-group", required=True, help="tags every record; resume key")
    ap.add_argument("--ids", nargs="+", help="only these instances (must be in the sample)")
    ap.add_argument("--max-instances", type=int, default=None)
    ap.add_argument("--stop-at", help="UTC time to stop starting new instances, e.g. 11:00")
    ap.add_argument("--daily-budget", type=int, default=950)
    ap.add_argument("--min-remaining", type=int, default=MIN_REMAINING)
    ap.add_argument("--min-free-gb", type=float, default=MIN_FREE_GB)
    ap.add_argument("--keep-images", action="store_true", help="do not delete pulled images")
    ap.add_argument("--dry-run", action="store_true", help="print the plan; no Docker, no calls")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    sample_path = ROOT / args.sample
    sample = json.loads(sample_path.read_text())
    ids = list(sample["instances"])
    if args.ids:
        unknown = set(args.ids) - set(ids)
        if unknown:
            raise SystemExit(f"not in {args.sample}: {sorted(unknown)}")
        ids = [i for i in ids if i in args.ids]
    env_path = ROOT / (args.env_record or args.sample.replace("_frozen", "_env"))
    if env_path == sample_path:
        raise SystemExit("--env-record must differ from the sample file")
    env = EnvRecord(env_path, sample.get("name", sample_path.stem))

    m11 = _load("paired_sweep_batch", "11_paired_sweep.py")
    per_turn, total = m11.default_budgets(args.model)
    cfg = m11.Config(model=args.model, arms="AB", run_group=args.run_group,
                     split="lite" if "Lite" in sample.get("dataset", "Lite") else "verified",
                     max_tokens_per_turn=per_turn, max_total_tokens=total,
                     daily_budget=args.daily_budget, min_remaining=args.min_remaining,
                     verbose=args.verbose)

    if args.dry_run:
        done = m11.completed_units(m11.RESULTS_FILE, m11.build_client(cfg.model).name,
                                   cfg.run_group, cfg.arms)
        todo = [i for i in ids if i not in done and env.state(i) in (None, "valid", "pending")]
        print(f"{len(ids)} in sample, {len(done & set(ids))} done, "
              f"{len(ids) - len(todo) - len(done & set(ids))} excluded, {len(todo)} to run")
        for i in todo:
            print(f"  {i}  env={env.state(i) or 'unchecked'}")
        return EXIT_DONE

    stop_at = next_utc(args.stop_at) if args.stop_at else None

    out = cfg.out_dir / cfg.run_group
    lock = out / "batch.lock"
    _lock(lock)
    log = (out / "batch.log").open("a", encoding="utf-8")
    real_out, real_err = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = m11.Tee(real_out, log), m11.Tee(real_err, log)
    try:
        print(f"\n{'#' * 64}\n# batch start {utcnow()}  model {cfg.model}  group {cfg.run_group}"
              f"\n# sample {args.sample} ({len(ids)} ids)  env {env_path.name}")
        from tdbf.data.instances import InstanceStore
        check = _load("build_sample_batch", "13_build_sample.py").check_instance
        b = run_batch(ids, store=InstanceStore(cfg.split), cfg=cfg, env=env, docker=Docker(),
                      m11=m11, check_instance=check, max_instances=args.max_instances,
                      stop_at=stop_at, min_remaining=args.min_remaining,
                      min_free_gb=args.min_free_gb, keep_images=args.keep_images)
        print(f"\n{'=' * 64}\nbatch end {utcnow()}: {len(b.done)} paired, "
              f"{len(b.excluded)} excluded, {len(b.failed)} failed, {b.skipped} skipped")
        for iid, why in b.excluded:
            print(f"  excluded {iid}: {why}")
        for iid, why in b.failed:
            print(f"  failed   {iid}: {why}  (retried next run)")
        if b.stop_reason:
            print(f"  STOPPED: {b.stop_reason}")
        return b.status
    finally:
        sys.stdout.flush()
        sys.stdout, sys.stderr = real_out, real_err
        log.close()
        lock.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
