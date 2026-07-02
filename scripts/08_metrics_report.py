#!/usr/bin/env python3
"""Metric report over results.jsonl — the numbers for the paper/advisor.

Pure aggregation of runs already logged (NO API, NO Docker). The metric logic
lives in `tdbf.metrics` (single source of truth, also used by the webapp Metrics
page); this script just loads results.jsonl and prints the text report.

  python3 scripts/08_metrics_report.py
  python3 scripts/08_metrics_report.py --model sonnet
"""

from __future__ import annotations

import argparse

from tdbf import metrics


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--file", default="results.jsonl")
    ap.add_argument("--model", help="filter to one model substring (e.g. sonnet)")
    args = ap.parse_args()

    recs = metrics.load_records(args.file)
    if args.model:
        recs = [r for r in recs if args.model in (r.get("model") or "")]

    print(metrics.format_text(metrics.build_report(recs)))


if __name__ == "__main__":
    main()
