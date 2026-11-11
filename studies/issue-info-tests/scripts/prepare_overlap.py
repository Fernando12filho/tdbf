#!/usr/bin/env python3
"""Step B input: the bugs of our 114 that the Bug Reports paper labelled as bug reports,
with the exact text they annotated (their cleaned SWE-bench Verified problem statement).

Writes ids.txt and texts.jsonl for split_issues.py, and human_spans.jsonl: the paper's
final (consensus) spans for those bugs, BUG_* labels only. The human spans are derived
from Bruno et al., arXiv:2607.09553, Zenodo 10.5281/zenodo.21035233, CC-BY-4.0.

    python3 prepare_overlap.py --package external/bug-reports-2607.09553/scripts/scripts \
        --sample sample_tests_swt114_frozen.json --out-dir studies/issue-info-tests/data
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--package", required=True)
    ap.add_argument("--sample", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--select", choices=("overlap", "dev"), default="overlap",
                    help="overlap: our bugs they labelled (the evaluation set); dev: a random sample "
                         "of their OTHER bug reports, for developing the labelling prompt")
    ap.add_argument("--n", type=int, default=40, help="dev sample size")
    ap.add_argument("--seed", type=int, default=20261111, help="dev sample seed")
    a = ap.parse_args()
    pkg, out = Path(a.package), Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    s = json.loads(Path(a.sample).read_text())
    ours = {i if isinstance(i, str) else i["instance_id"] for i in s["instances"]}
    bug_reports = {json.loads(l)["instance_id"] for l in (pkg / "inter_annotator_agreement_labels.jsonl").read_text().splitlines() if l}
    clean = {r["instance_id"]: r for r in map(json.loads, (pkg / "data/clean-dataset/clean_dataset.jsonl").read_text().splitlines()) if r}
    if a.select == "overlap":
        ids = sorted(i for i in ours if i in bug_reports)
    else:
        import random
        pool = sorted(bug_reports - ours)
        ids = sorted(random.Random(a.seed).sample(pool, a.n))

    (out / "overlap_ids.txt").write_text("\n".join(ids) + "\n")
    with (out / "overlap_texts.jsonl").open("w") as f:
        for i in ids:
            f.write(json.dumps({"instance_id": i, "text": clean[i]["problem_statement"],
                                "duplicated_in_source": bool(clean[i].get("duplicated_statement"))}) + "\n")
    ann = json.loads((pkg / "data/annotations/annotations.json").read_text())
    with (out / "human_spans.jsonl").open("w") as f:
        for a_ in ann:
            if a_["instance_id"] in ids and a_["label"].startswith("BUG_"):
                f.write(json.dumps({k: a_[k] for k in ("instance_id", "label", "start", "end")}) + "\n")
    iaa = [json.loads(l) for l in (pkg / "inter_annotator_agreement_labels.jsonl").read_text().splitlines() if l]
    with (out / "human_presence_by_annotator.jsonl").open("w") as f:
        for r in iaa:
            if r["instance_id"] in ids:
                f.write(json.dumps(r) + "\n")
    (out / "SOURCE.txt").write_text(
        "human_spans.jsonl and human_presence_by_annotator.jsonl are derived from the replication\n"
        "package of V. L. Bruno, A. Giagnorio, D. Bifolco, L. Wienges, M. Di Penta, G. Bavota,\n"
        "'Writing Bug Reports for Software Repair Agents: What Information Matters Most?',\n"
        "arXiv:2607.09553, Zenodo doi:10.5281/zenodo.21035233, licensed CC-BY-4.0.\n"
        "Only the rows for the bugs listed in overlap_ids.txt are kept; offsets refer to the\n"
        "texts in overlap_texts.jsonl (their cleaned SWE-bench Verified problem statements).\n")
    if a.select == "dev":
        assert not set(ids) & ours, "dev sample must not contain evaluation bugs"
        print(f"{len(ids)} dev bugs (seed {a.seed}) from {len(bug_reports - ours)} labelled bug reports outside our sample")
    else:
        print(f"{len(ids)} bugs; not bug reports in their study: {sorted(ours & set(clean) - set(ids))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
