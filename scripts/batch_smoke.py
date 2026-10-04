"""Offline checks for scripts/16_nightly_batch.py. No Docker, no API.

Fake Docker, a fake paired sweep and a fake gold check drive the real
run_batch through every branch: pull + tag + delete, a local image kept,
pull failure, gold failure, already-done and already-excluded skips, a valid
instance re-run after a failed unit, quota / disk / failure stops, and the
environment record it writes.

    python3 scripts/batch_smoke.py
"""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _load():
    spec = importlib.util.spec_from_file_location("nightly_batch", ROOT / "scripts" /
                                                  "16_nightly_batch.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


B = _load()


class FakeDocker:
    def __init__(self, local=(), unpullable=(), free=100.0):
        self.images = {B.short_image(i) for i in local}
        self.unpullable, self.free = set(unpullable), free
        self.pulled, self.removed = [], []

    def has(self, image):
        return image in self.images

    def free_gb(self):
        return self.free

    def pull(self, image):
        if any(image == B.registry_image(i) for i in self.unpullable):
            return False, "manifest unknown"
        self.images.add(image)
        self.pulled.append(image)
        return True, ""

    def tag(self, src, dst):
        self.images.add(dst)
        return True

    def digest(self, image):
        return "swebench/x@sha256:abc"

    def remove(self, *images):
        for im in images:
            self.images.discard(im)
        self.removed.append(images)


class FakeUsage:
    left = 950

    def __init__(self, path, budget):
        pass

    def remaining(self):
        return FakeUsage.left

    def used_today(self):
        return 950 - FakeUsage.left


def fake_m11(done=(), unit_results=None):
    """unit_results: iid -> 'done' | 'fail' | 'quota' (default 'done')."""
    ran = []

    def run_sweep(cfg, instances):
        iid = instances[0].instance_id
        ran.append(iid)
        r = (unit_results or {}).get(iid, "done")
        if r == "quota":
            return SimpleNamespace(status=3, stop_reason="429", done=[], failed=[])
        if r == "fail":
            return SimpleNamespace(status=0, stop_reason=None, done=[], failed=[(iid, "503")])
        return SimpleNamespace(status=0, stop_reason=None, done=[{"instance_id": iid}], failed=[])

    m = SimpleNamespace(Usage=FakeUsage, EXIT_QUOTA=3, RESULTS_FILE=Path("unused"),
                        build_client=lambda spec, **kw: SimpleNamespace(name=spec),
                        completed_units=lambda *a: set(done), run_sweep=run_sweep)
    return m, ran


def fake_check(invalid=()):
    checked = []

    def check(inst, split, run_dir):
        checked.append(inst.instance_id)
        ok = inst.instance_id not in invalid
        return {"valid": ok, "gold_resolved": ok, "gold_error": None,
                "brt_error": None, "checked_utc": "2026-10-05T01:00:00+00:00"}
    return check, checked


STORE = SimpleNamespace(get=lambda iid: SimpleNamespace(instance_id=iid))


def batch(ids, td, *, docker, m11, check, env=None, **kw):
    cfg = SimpleNamespace(model="m", run_group="g", split="lite", arms="AB", num_ctx=None,
                          daily_budget=950, out_dir=Path(td))
    env = env or B.EnvRecord(Path(td) / "env.json", "s")
    kw.setdefault("max_instances", None)
    kw.setdefault("stop_at", None)
    kw.setdefault("min_remaining", 140)
    kw.setdefault("min_free_gb", 25.0)
    return B.run_batch(ids, store=STORE, cfg=cfg, env=env, docker=docker, m11=m11,
                       check_instance=check, **kw), env


def main() -> int:
    with tempfile.TemporaryDirectory() as td:
        FakeUsage.left = 950
        d = FakeDocker(local=["l__l-1"], unpullable=["u__u-1"])
        m11, ran = fake_m11(done=["d__d-1"])
        check, checked = fake_check(invalid=["g__g-1"])
        env = B.EnvRecord(Path(td) / "env.json", "s")
        env.set("x__x-1", state="environment_invalid", reason="earlier")
        b, env = batch(["p__p-1", "l__l-1", "u__u-1", "g__g-1", "d__d-1", "x__x-1"], td,
                       docker=d, m11=m11, check=check, env=env)
        assert b.done == ["p__p-1", "l__l-1"] and b.status == B.EXIT_DONE, b
        assert ran == ["p__p-1", "l__l-1"], ran                     # no unit for u/g/d/x
        assert checked == ["p__p-1", "l__l-1", "g__g-1"], checked    # gold before arms
        assert b.skipped == 2 and sorted(b.excluded) == [
            ("g__g-1", "environment_invalid"), ("u__u-1", "image_unavailable")]
        # pulled images are deleted; the local image is kept; the failed pull left nothing
        assert d.pulled == [B.registry_image("p__p-1"), B.registry_image("g__g-1")], d.pulled
        assert {B.short_image("p__p-1"), B.short_image("g__g-1")}.isdisjoint(d.images)
        assert B.short_image("l__l-1") in d.images
        rec = json.loads((Path(td) / "env.json").read_text())["instances"]
        assert rec["p__p-1"]["state"] == "valid" and "sha256" in rec["p__p-1"]["image"]
        assert rec["l__l-1"]["image"] == "local image (already present)"
        assert rec["u__u-1"]["state"] == "image_unavailable"
        assert rec["g__g-1"]["state"] == "environment_invalid"
        assert rec["x__x-1"]["reason"] == "earlier"
        print("  ok  pull/tag/delete, local kept, pull + gold failures excluded before arms")

    with tempfile.TemporaryDirectory() as td:
        # a valid instance re-run after a failed unit: no second gold check, stays valid
        env = B.EnvRecord(Path(td) / "env.json", "s")
        env.set("v__v-1", state="valid", gold_resolved=True)
        m11, ran = fake_m11()
        check, checked = fake_check()
        b, env = batch(["v__v-1"], td, docker=FakeDocker(), m11=m11, check=check, env=env)
        assert b.done == ["v__v-1"] and checked == [] and env.state("v__v-1") == "valid"
        print("  ok  re-run of a valid instance skips the gold check and stays valid")

    with tempfile.TemporaryDirectory() as td:
        m11, ran = fake_m11(unit_results={i: "fail" for i in ("a__a-1", "a__a-2", "a__a-3")})
        d = FakeDocker()
        b, _ = batch(["a__a-1", "a__a-2", "a__a-3", "a__a-4"], td, docker=d, m11=m11,
                     check=fake_check()[0])
        assert b.status == B.EXIT_FAILURES and ran == ["a__a-1", "a__a-2", "a__a-3"]
        assert len(d.removed) == 3                     # images go even when a unit fails
        print("  ok  three failed units in a row stop the batch; images still removed")

    with tempfile.TemporaryDirectory() as td:
        m11, ran = fake_m11(unit_results={"q__q-1": "quota"})
        d = FakeDocker()
        b, _ = batch(["q__q-1", "q__q-2"], td, docker=d, m11=m11, check=fake_check()[0])
        assert b.status == B.EXIT_QUOTA and ran == ["q__q-1"] and len(d.removed) == 1

        FakeUsage.left = 100                            # < 140: stop before pulling
        d = FakeDocker()
        b, _ = batch(["q__q-2"], td, docker=d, m11=fake_m11()[0], check=fake_check()[0])
        assert b.status == B.EXIT_QUOTA and d.pulled == []
        FakeUsage.left = 950

        d = FakeDocker(free=10.0)
        b, _ = batch(["q__q-2"], td, docker=d, m11=fake_m11()[0], check=fake_check()[0])
        assert b.status == B.EXIT_DISK and d.pulled == []

        b, _ = batch(["q__q-2", "q__q-3"], td, docker=FakeDocker(), m11=fake_m11()[0],
                     check=fake_check()[0], max_instances=1)
        assert b.done == ["q__q-2"] and b.status == B.EXIT_DONE

        past = datetime(2020, 1, 1, tzinfo=timezone.utc)
        b, _ = batch(["q__q-3"], td, docker=FakeDocker(), m11=fake_m11()[0],
                     check=fake_check()[0], stop_at=past)
        assert b.status == B.EXIT_TIME and not b.done
        print("  ok  quota (mid-unit and up front), disk, --max-instances, --stop-at")

    now = datetime(2026, 10, 5, 22, 0, tzinfo=timezone.utc)
    assert B.next_utc("23:30", now) == datetime(2026, 10, 5, 23, 30, tzinfo=timezone.utc)
    assert B.next_utc("11:00", now) == datetime(2026, 10, 6, 11, 0, tzinfo=timezone.utc)
    assert B.registry_image("django__django-11099") == \
        "swebench/sweb.eval.x86_64.django_1776_django-11099:latest"
    print("  ok  next_utc rolls to tomorrow; registry image name")
    print("\nNightly batch: all checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
