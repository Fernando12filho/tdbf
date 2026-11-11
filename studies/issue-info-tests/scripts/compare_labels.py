#!/usr/bin/env python3
"""Step B analysis: how well do LLM labels agree with the human labels?

Three comparisons on the same bugs, reported side by side:
  human A vs human B   the paper's two annotators (presence per label, per issue)
  LLM vs human final   the LLM against the paper's consensus labels
  LLM run vs LLM run   the LLM against itself (consistency across runs)

Measures, as in Bruno et al. (arXiv:2607.09553):
  presence  for each label, is it present in the issue (yes/no)? Cohen's kappa over
            issues, kappa = (p_o - p_e) / (1 - p_e), plus raw agreement p_o.
  characters each character's label set (empty = unlabelled); agreement = share of
            characters where the two sets are equal.
Also reported for the macro-categories of the paper's ablation (location = any
AFFECTED_AREA, suggestion = any SUGGESTED_FIX, behavioural context = EXPECTED or
REPRODUCTION) and for BUG_EXPECTED_BEHAVIOUR alone, our variable of interest.

    python3 compare_labels.py --data studies/issue-info-tests/data \
        --llm studies/issue-info-tests/results/llm_spans_gpt-5-mini.jsonl
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

MACRO = {
    "location": lambda l: l.startswith("BUG_AFFECTED_AREA"),
    "suggestion": lambda l: l.startswith("BUG_SUGGESTED_FIX"),
    "behavioural_context": lambda l: l in ("BUG_EXPECTED_BEHAVIOUR", "BUG_REPRODUCTION_STEPS"),
    "expected_behaviour": lambda l: l == "BUG_EXPECTED_BEHAVIOUR",
    "observed_behaviour": lambda l: l == "BUG_OBSERVED_BEHAVIOUR",
    "reproduction_steps": lambda l: l == "BUG_REPRODUCTION_STEPS",
}


def kappa(a: list[int], b: list[int]) -> tuple[float | None, float]:
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return (None if pe == 1 else (po - pe) / (1 - pe)), po


def presence(spans: list[dict], ids: list[str], test) -> list[int]:
    have = {s["instance_id"] for s in spans if test(s["label"])}
    return [int(i in have) for i in ids]


def char_agreement(a: list[dict], b: list[dict], texts: dict[str, str]) -> float:
    agree = total = 0
    for iid, text in texts.items():
        la = [set() for _ in text]
        lb = [set() for _ in text]
        for spans, lab in ((a, la), (b, lb)):
            for s in spans:
                if s["instance_id"] == iid:
                    for c in range(s["start"], min(s["end"], len(text))):
                        lab[c].add(s["label"])
        agree += sum(x == y for x, y in zip(la, lb))
        total += len(text)
    return agree / total


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--llm", required=True, nargs="+", help="one or more span files (parallel parts)")
    a = ap.parse_args()
    d = Path(a.data)
    ids = [i.strip() for i in (d / "overlap_ids.txt").read_text().splitlines() if i.strip()]
    texts = {r["instance_id"]: r["text"] for r in map(json.loads, (d / "overlap_texts.jsonl").read_text().splitlines())}
    human = [json.loads(l) for l in (d / "human_spans.jsonl").read_text().splitlines() if l]
    iaa = {r["instance_id"]: r for r in map(json.loads, (d / "human_presence_by_annotator.jsonl").read_text().splitlines())}
    recs = [json.loads(l) for p in a.llm for l in Path(p).read_text().splitlines() if l]
    keys = [(r["instance_id"], r["run"]) for r in recs]
    if len(keys) != len(set(keys)):
        raise SystemExit("duplicate (issue, run) records across the span files")
    runs = collections.defaultdict(list)
    errors = [r for r in recs if r.get("error")]
    for r in recs:
        for s in r["spans"]:
            runs[r["run"]].append({"instance_id": r["instance_id"], **s})
    unmatched = sum(len(r["unmatched"]) for r in recs)
    matched = sum(len(r["spans"]) for r in recs)
    cost = sum(float((r.get("usage") or {}).get("cost") or 0) for r in recs)
    print(f"LLM records {len(recs)} (errors {len(errors)}), spans placed {matched}, not verbatim/unknown {unmatched}, cost ${cost:.3f}")
    labels = sorted({s["label"] for s in human} | {s["label"] for r in runs.values() for s in r})

    def annot(n: int, test) -> list[int]:
        out = []
        for i in ids:
            r = iaa[i]
            out.append(int(any(v for k, v in r.items() if k.startswith(f"annotator{n}_BUG") and test(k.split("_", 1)[1]))))
        return out

    print("\nPresence per issue: Cohen's kappa (raw agreement), n =", len(ids))
    print(f"{'category':34} {'human A vs B':>16} {'LLM run1 vs human':>18} {'LLM run1 vs run2':>17} {'human/LLM present':>18}")
    rows = [(l, (lambda lab: (lambda x: x == lab))(l)) for l in labels] + [(f"[macro] {m}", t) for m, t in MACRO.items()]
    for name, test in rows:
        h = presence(human, ids, test)
        l1 = presence(runs[1], ids, test)
        l2 = presence(runs[2], ids, test) if 2 in runs else None
        hab = kappa(annot(1, test), annot(2, test))
        lh = kappa(l1, h)
        ll = kappa(l1, l2) if l2 else (None, float("nan"))
        f = lambda k: ("  n/a " if k[0] is None else f"{k[0]:6.2f}") + f" ({k[1]:.2f})"
        print(f"{name:34} {f(hab):>16} {f(lh):>18} {f(ll):>17} {sum(h):>8}/{sum(l1):<8}")
    for r in sorted(runs):
        print(f"\ncharacter agreement, LLM run {r} vs human final: {char_agreement(runs[r], human, texts):.3f}")
    if 2 in runs:
        print(f"character agreement, LLM run 1 vs run 2: {char_agreement(runs[1], runs[2], texts):.3f}")
    print("(paper, human A vs human B on all 441 issues: 78.6% of characters, presence kappa 0.57)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
