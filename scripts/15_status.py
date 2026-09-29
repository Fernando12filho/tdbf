"""Where every SWE-bench Lite instance stands: run, pending, excluded, or never drawn.

Reads results.jsonl, the frozen sample and its environment record, and asks
Docker which instance images are on disk. No model calls, writes nothing except
the Lite id cache (runs/lite_index.json) on first use. The webapp Status page
shows the same table (both use tdbf.ledger).

    python3 scripts/15_status.py                    # summary by status and repo
    python3 scripts/15_status.py --list unused      # ids with one status
    python3 scripts/15_status.py --list excluded -v # ... with the reason
    python3 scripts/15_status.py --no-docker        # skip the image check
    python3 scripts/15_status.py --refresh-index    # re-read the Lite id list
    python3 scripts/15_status.py --json             # the whole table
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tdbf import ledger  # noqa: E402

SHORT = {"paired": "paired", "one_arm": "one arm", "pending": "pending",
         "excluded": "excluded", "development": "dev set", "unused": "never drawn"}


def print_summary(led: dict) -> None:
    print(f"SWE-bench Lite: {led['total']} instances"
          + ("" if led["complete_index"] else
             "  (Lite index not cached: only instances seen in a sample or a run)"))
    print(f"Experiment arms: {', '.join(led['arms'])}\n")
    for key, label, meaning in ledger.STATUSES:
        print(f"  {led['counts'][key]:4d}  {label:20s} {meaning}")

    cols = list(ledger.STATUS_KEYS)
    head = ["repo", "total", *[SHORT[k] for k in cols]]
    if led["images_known"]:
        head.append("image")
    widths = [max(len(h), 26 if i == 0 else 0) for i, h in enumerate(head)]
    print("\n" + "  ".join(h.ljust(w) if i == 0 else h.rjust(max(w, 5))
                           for i, (h, w) in enumerate(zip(head, widths))))
    for t in led["repos"]:
        vals = [t["repo"], t["total"], *[t[k] for k in cols]]
        if led["images_known"]:
            vals.append(t["images"])
        print("  ".join(str(v).ljust(w) if i == 0 else str(v).rjust(max(w, 5))
                        for i, (v, w) in enumerate(zip(vals, widths))))

    if led["images_known"]:
        print(f"\nInstance images on disk: {led['images_on_disk']}")
        if led["images_orphaned"]:
            print(f"  not in the table (not a Lite id?): {', '.join(led['images_orphaned'])}")
    else:
        print("\nInstance images: unknown (Docker not reachable, or --no-docker)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", choices=ledger.STATUS_KEYS, metavar="STATUS",
                    help=f"print the ids with this status ({', '.join(ledger.STATUS_KEYS)})")
    ap.add_argument("--repo", help="with --list: only this repo, e.g. sympy/sympy")
    ap.add_argument("-v", "--verbose", action="store_true",
                    help="with --list: add arms done, environment state and reason")
    ap.add_argument("--json", action="store_true", help="print the full table as JSON")
    ap.add_argument("--no-docker", action="store_true", help="do not ask Docker for images")
    ap.add_argument("--refresh-index", action="store_true",
                    help="re-read the Lite id list from HuggingFace")
    args = ap.parse_args()

    if args.refresh_index or ledger.load_lite_index(ROOT) is None:
        try:
            ids = ledger.write_lite_index(ROOT)
            print(f"cached {len(ids)} Lite ids -> {ledger.LITE_INDEX}", file=sys.stderr)
        except Exception as exc:  # noqa: BLE001 - the table still works without it
            print(f"[warn] could not load the Lite dataset ({type(exc).__name__}: {exc}); "
                  "showing only instances seen in a sample or a run", file=sys.stderr)

    led = ledger.load_ledger(ROOT, docker=not args.no_docker)

    if args.json:
        print(json.dumps(led, indent=2))
    elif args.list:
        for r in led["rows"]:
            if r["status"] != args.list or (args.repo and r["repo"] != args.repo):
                continue
            line = r["instance_id"]
            if args.verbose:
                img = {True: "image", False: "no image", None: "image ?"}[r["image"]]
                line += (f"  arms={'+'.join(r['arms_done']) or '-'}  env={r['env_state'] or '-'}"
                         f"  runs={r['runs']}  {img}")
                if r["env_reason"]:
                    line += f"\n    {r['env_reason']}"
            print(line)
    else:
        print_summary(led)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
