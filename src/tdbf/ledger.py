"""Where every SWE-bench Lite instance stands — one row per instance.

Answers "what has been run, what is left, and is its image on disk?" for all
300 Lite instances at once, from files we already keep:

  - results.jsonl                    which runs exist (the experiment's arms are
                                     `metrics.EXPERIMENT_ARMS`, never model alone)
  - sample_lite_frozen.json          which instances are in the frozen sample, and
                                     which were the development set
  - sample_lite_env.json             each sample instance's environment state
  - runs/lite_index.json             the full Lite id list (a cache of the
                                     HuggingFace dataset, written by script 15)
  - `docker images`                  which instance images are on disk (optional)

Shared by scripts/15_status.py and the webapp Status page so the two cannot
disagree. Pure apart from the small loaders at the bottom.
"""

from __future__ import annotations

import collections
import json
import os
import subprocess
from pathlib import Path

from .metrics import EXPERIMENT_ARMS, load_records, unit_arms

# (frozen sample file, its environment record). An extension sample is added
# here once it is frozen; the development set is read from the first file.
SAMPLE_FILES = (
    ("sample_lite_frozen.json", "sample_lite_env.json"),
    ("sample_lite_ext_frozen.json", "sample_lite_ext_env.json"),   # extension, n=90
)

LITE_INDEX = Path("runs") / "lite_index.json"

# Display order, label, and what each status means. The webapp and the script
# both render from this, so the wording lives in one place.
STATUSES = (
    ("paired", "Paired",
     "In the sample; both arms finished. This is a usable RQ1 result."),
    ("one_arm", "One arm",
     "In the sample; only one arm finished. Needs the other arm."),
    ("pending", "Pending",
     "In the sample with a valid environment, but no arm has finished yet."),
    ("excluded", "Excluded / deferred",
     "In the sample, but its environment is not marked valid (no image, gold "
     "check failed, or dropped after provider errors). See the reason."),
    ("development", "Development set",
     "Used while building the agents. Kept out of every sample."),
    ("unused", "Never drawn",
     "Not in any sample and not in the development set. Available for an "
     "extension sample."),
)
STATUS_KEYS = tuple(k for k, _, _ in STATUSES)

def repo_from_id(instance_id: str) -> str:
    """'django__django-11620' -> 'django/django'. SWE-bench ids are built this way."""
    owner, sep, rest = instance_id.partition("__")
    return f"{owner}/{rest.rsplit('-', 1)[0]}" if sep else "?"


def image_name(instance_id: str) -> str:
    return f"sweb.eval.x86_64.{instance_id}:latest"


# ── the table ─────────────────────────────────────────────────────────────────

def build_ledger(*, lite_ids: list[str] | None, records: list[dict],
                 samples: list[dict], env_states: dict[str, dict],
                 images: set[str] | None) -> dict:
    """One row per instance plus summaries.

    `lite_ids` None means the Lite index is not cached yet: the table then only
    covers instances that appear in a sample or in results.jsonl, and says so.
    `samples` are the parsed frozen-sample files; `env_states` maps instance id
    to its environment-record row ({"state", "reason", ...}); `images` is the set
    of instance ids with a local image, or None when Docker could not be asked.
    """
    dev = set(samples[0].get("excluded_development_set") or []) if samples else set()
    sample_of: dict[str, str] = {}
    for s in samples:
        for iid in s.get("instances") or []:
            sample_of.setdefault(iid, s.get("name", "sample"))

    # Which experiment arms finished, from sweep_unit records only: a unit is
    # written only after its arm(s) produced a verified verdict.
    done: dict[str, set[str]] = collections.defaultdict(set)
    runs: dict[str, int] = collections.Counter()
    models: dict[str, set[str]] = collections.defaultdict(set)
    last: dict[str, str] = {}
    for r in records:
        iid = r.get("instance_id")
        if not iid:
            continue
        if r.get("mode") != "sweep_unit":
            runs[iid] += 1
            if r.get("model"):
                models[iid].add(r["model"])
        ts = r.get("timestamp") or ""
        if ts > last.get(iid, ""):
            last[iid] = ts
        done[iid] |= unit_arms(r)

    ids = set(lite_ids or []) | set(sample_of) | dev | set(runs)
    n_arms = len(EXPERIMENT_ARMS)
    rows = []
    for iid in sorted(ids):
        env = env_states.get(iid) or {}
        arms = done.get(iid, set())
        if iid in dev:
            status = "development"
        elif iid not in sample_of:
            status = "unused"
        elif len(arms) == n_arms:
            status = "paired"
        elif env.get("state") not in (None, "valid"):
            status = "excluded"
        elif arms:
            status = "one_arm"
        else:
            status = "pending"
        rows.append({
            "instance_id": iid,
            "repo": repo_from_id(iid),
            "status": status,
            "sample": sample_of.get(iid),
            "arms_done": sorted(arms),
            "env_state": env.get("state"),
            "env_reason": env.get("reason"),
            "gold_resolved": env.get("gold_resolved"),
            "runs": runs.get(iid, 0),
            "models": sorted(models.get(iid, ())),
            "last_run": (last.get(iid) or "")[:10] or None,
            "image": None if images is None else iid in images,
        })

    by_status = collections.Counter(r["status"] for r in rows)
    repos: dict[str, dict] = {}
    for r in rows:
        t = repos.setdefault(r["repo"], {"repo": r["repo"], "total": 0, "images": 0,
                                         **{k: 0 for k in STATUS_KEYS}})
        t["total"] += 1
        t[r["status"]] += 1
        t["images"] += bool(r["image"])
    return {
        "rows": rows,
        "counts": {k: by_status.get(k, 0) for k in STATUS_KEYS},
        "total": len(rows),
        "complete_index": lite_ids is not None,
        "repos": sorted(repos.values(), key=lambda t: (-t["total"], t["repo"])),
        "arms": [label for label, _, _ in EXPERIMENT_ARMS],
        "images_known": images is not None,
        "images_on_disk": None if images is None else len(images),
        "images_orphaned": (None if images is None
                            else sorted(images - ids)),
    }


# ── loaders ───────────────────────────────────────────────────────────────────

def load_lite_index(root: Path) -> list[str] | None:
    path = root / LITE_INDEX
    try:
        return json.loads(path.read_text())["instances"]
    except (OSError, ValueError, KeyError):
        return None


def write_lite_index(root: Path, split: str = "lite") -> list[str]:
    """Cache the Lite id list from HuggingFace (needs `datasets` and the
    network or a warm HF cache). Everything else here works offline."""
    from .data.instances import InstanceStore

    store = InstanceStore(split)
    ids = sorted(i.instance_id for i in store.iter_instances())
    path = root / LITE_INDEX
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"dataset": store.dataset_name, "instances": ids}, indent=1))
    return ids


def load_samples(root: Path) -> tuple[list[dict], dict[str, dict]]:
    samples, env = [], {}
    for sample_file, env_file in SAMPLE_FILES:
        try:
            samples.append(json.loads((root / sample_file).read_text()))
        except (OSError, ValueError):
            continue
        try:
            env.update(json.loads((root / env_file).read_text())["instances"])
        except (OSError, ValueError, KeyError):
            pass
    return samples, env


def local_images(timeout: float = 10.0) -> set[str] | None:
    """Instance ids that have a `sweb.eval.x86_64.<id>:latest` image locally,
    or None if Docker is not reachable (Colima stopped, no daemon)."""
    env = dict(os.environ)
    sock = Path.home() / ".colima" / "default" / "docker.sock"
    if not env.get("DOCKER_HOST") and sock.exists():
        env["DOCKER_HOST"] = f"unix://{sock}"
    try:
        out = subprocess.run(
            ["docker", "images", "--format", "{{.Repository}}:{{.Tag}}"],
            capture_output=True, text=True, timeout=timeout, env=env)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if out.returncode != 0:
        return None
    prefix, suffix = "sweb.eval.x86_64.", ":latest"
    return {line[len(prefix):-len(suffix)] for line in out.stdout.split()
            if line.startswith(prefix) and line.endswith(suffix)}


def load_ledger(root: Path, *, docker: bool = True) -> dict:
    """Everything above, read from the repo at `root`."""
    samples, env = load_samples(root)
    return build_ledger(
        lite_ids=load_lite_index(root),
        records=load_records(root / "results.jsonl"),
        samples=samples,
        env_states=env,
        images=local_images() if docker else None,
    )
