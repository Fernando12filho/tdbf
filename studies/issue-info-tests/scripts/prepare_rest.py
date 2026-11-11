#!/usr/bin/env python3
"""Inputs for labelling the bugs of our 114 that have no human labels.

Takes the issue text from SWE-bench Lite (the dataset our runs used). Some
problem statements in SWE-bench contain the same text twice in a row; the Bug Reports
paper removed the duplicate before labelling, and we do the same so that both label
sets are made on comparable text. Writes rest_ids.txt and rest_texts.jsonl.

    python3 prepare_rest.py --sample sample_tests_swt114_frozen.json \
        --overlap studies/issue-info-tests/data/overlap_ids.txt --out-dir studies/issue-info-tests/data
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def dedupe(text: str) -> tuple[str, bool]:
    """If the text is one block repeated twice (ignoring surrounding whitespace), keep one."""
    t = text.strip()
    n = len(t)
    for cut in range(n // 2 - 2, n // 2 + 3):
        if 0 < cut < n and t[:cut].strip() and t[:cut].strip() == t[cut:].strip():
            return t[:cut].strip(), True
    return text, False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", required=True)
    ap.add_argument("--overlap", required=True)
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from datasets import load_dataset
    lite = {r["instance_id"]: r for r in load_dataset("princeton-nlp/SWE-bench_Lite", split="test")}
    s = json.loads(Path(a.sample).read_text())
    ours = sorted(i if isinstance(i, str) else i["instance_id"] for i in s["instances"])
    overlap = {l.strip() for l in Path(a.overlap).read_text().splitlines() if l.strip()}
    ids = [i for i in ours if i not in overlap]
    out = Path(a.out_dir)
    (out / "rest_ids.txt").write_text("\n".join(ids) + "\n")
    dups = []
    with (out / "rest_texts.jsonl").open("w") as f:
        for i in ids:
            text, dup = dedupe(lite[i]["problem_statement"])
            if dup:
                dups.append(i)
            f.write(json.dumps({"instance_id": i, "text": text, "duplicated_in_source": dup}) + "\n")
    print(f"{len(ids)} bugs without human labels; duplicated texts cleaned: {dups}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
