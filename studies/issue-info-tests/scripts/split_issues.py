#!/usr/bin/env python3
"""Step B: an LLM splits issue texts into the Bug Reports paper's information types.

The LLM marks spans of the ORIGINAL text; it never rewrites it. Each returned span must
be a verbatim substring of the issue and is mapped back to character offsets, so the
output has the same shape as the human annotations of Bruno et al. (arXiv:2607.09553,
Zenodo 10.5281/zenodo.21035233, CC-BY-4.0): (instance_id, label, start, end).

Label names and definitions are read from that package's labels.json (bug labels only).
Each issue is labelled `--runs` times so the LLM's own consistency can be reported.

    python3 split_issues.py --ids ids.txt --texts texts.jsonl \
        --labels external/.../labels.json --env /path/.env \
        --model openai/gpt-5-mini --runs 3 --out results/llm_spans.jsonl

texts.jsonl: one {"instance_id", "text"} per line. Output: one record per (issue, run)
with spans, unmatched spans, token usage and cost. Existing (issue, run) pairs in --out
are skipped, so an interrupted run resumes.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

API = "https://openrouter.ai/api/v1/chat/completions"

PROMPT = """You annotate a GitHub bug report. Mark the parts of the text that convey each \
type of information below. Copy every span VERBATIM from the report: exact characters, \
including code, punctuation and line breaks. Do not paraphrase, summarise or fix typos. \
A span can be a whole sentence, part of a sentence, or a code block. Leave text that \
conveys none of these types unmarked (greetings, frustration, template headings). \
The same text may be marked with more than one type only when it clearly conveys both.

Information types:
{labels}

Return only JSON, no commentary, in this form:
{{"spans": [{{"label": "<one of the type names above>", "text": "<verbatim span>"}}]}}

Bug report:
<<<REPORT
{text}
REPORT>>>"""


def read_key(env_path: Path) -> str:
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if line.startswith("OPENROUTER_API_KEY="):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    sys.exit(f"OPENROUTER_API_KEY not found in {env_path}")


class Deadline(Exception):
    pass


def _alarm(signum, frame):
    raise Deadline("no complete reply within the per-call deadline")


def call(key: str, model: str, prompt: str, retries: int = 3, deadline: int = 240) -> dict:
    """One completion, with a hard wall-clock deadline per attempt.

    The socket timeout alone is not enough: OpenRouter keeps a slow request alive by
    sending whitespace while the model works, which resets the socket timer, so a
    stalled call could hang for over an hour (seen 2026-11-11). SIGALRM bounds it."""
    import signal
    body = json.dumps({"model": model, "messages": [{"role": "user", "content": prompt}],
                       "usage": {"include": True}}).encode()
    signal.signal(signal.SIGALRM, _alarm)
    for attempt in range(retries):
        req = urllib.request.Request(API, data=body, headers={
            "Authorization": f"Bearer {key}", "Content-Type": "application/json",
            "X-Title": "tdbf issue-info-tests step B"})
        signal.alarm(deadline)
        try:
            return json.load(urllib.request.urlopen(req, timeout=120))
        except (urllib.error.URLError, TimeoutError, Deadline) as e:
            if attempt == retries - 1:
                raise
            print(f"  retry after {e}", file=sys.stderr, flush=True)
            time.sleep(5 * (attempt + 1))
        finally:
            signal.alarm(0)
    raise RuntimeError("unreachable")


def parse_spans(content: str) -> list[dict]:
    m = re.search(r"\{.*\}", content, re.S)
    if not m:
        raise ValueError("no JSON object in the reply")
    # strict=False: models sometimes copy a raw tab or newline from the issue into a
    # JSON string; that is still a verbatim span (seen on django-13807, dev round v2).
    return json.loads(m.group(0), strict=False).get("spans", [])


def locate(text: str, span: str, used: set[tuple[int, int]]) -> tuple[int, int] | None:
    """Offsets of `span` in `text`: exact first, then ignoring whitespace differences.
    Prefers an occurrence not already used by the same label."""
    starts = [m.start() for m in re.finditer(re.escape(span), text)] if span else []
    for s in starts:
        if (s, s + len(span)) not in used:
            return s, s + len(span)
    if starts:
        return starts[0], starts[0] + len(span)
    pattern = r"\s+".join(re.escape(tok) for tok in span.split())
    if pattern:
        m = re.search(pattern, text)
        if m:
            return m.start(), m.end()
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ids", required=True)
    ap.add_argument("--texts", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--env", required=True)
    ap.add_argument("--model", default="openai/gpt-5-mini")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--out", required=True)
    ap.add_argument("--prompt", help="prompt template file with {labels} and {text}; default = built-in v1")
    a = ap.parse_args()

    import hashlib
    template = Path(a.prompt).read_text() if a.prompt else PROMPT
    prompt_sha = hashlib.sha256(template.encode()).hexdigest()[:12]
    labels = [l for l in json.loads(Path(a.labels).read_text()) if l["name"].startswith("BUG_")]
    names = {l["name"] for l in labels}
    label_text = "\n".join(f"- {l['name']}: {l['description']}" for l in labels)
    texts = {r["instance_id"]: r["text"] for r in map(json.loads, Path(a.texts).read_text().splitlines()) if r}
    ids = [i.strip() for i in Path(a.ids).read_text().splitlines() if i.strip()]
    out = Path(a.out)
    done = set()
    if out.exists():
        for r in map(json.loads, out.read_text().splitlines()):
            done.add((r["instance_id"], r["run"]))
    key = read_key(Path(a.env))

    total = 0.0
    for iid in ids:
        text = texts[iid]
        for run in range(1, a.runs + 1):
            if (iid, run) in done:
                continue
            resp = call(key, a.model, template.format(labels=label_text, text=text))
            content = resp["choices"][0]["message"].get("content") or ""
            usage = resp.get("usage") or {}
            rec = {"instance_id": iid, "run": run, "model": a.model, "prompt_sha": prompt_sha,
                   "served_model": resp.get("model"), "provider": resp.get("provider"),
                   "usage": usage, "spans": [], "unmatched": [], "error": None}
            try:
                used: dict[str, set] = {}
                for s in parse_spans(content):
                    lab, span = s.get("label"), s.get("text") or ""
                    if lab not in names:
                        rec["unmatched"].append({"label": lab, "text": span, "why": "unknown label"})
                        continue
                    pos = locate(text, span, used.setdefault(lab, set()))
                    if pos is None:
                        rec["unmatched"].append({"label": lab, "text": span, "why": "not verbatim"})
                        continue
                    used[lab].add(pos)
                    rec["spans"].append({"label": lab, "start": pos[0], "end": pos[1]})
            except (ValueError, json.JSONDecodeError) as e:
                rec["error"] = str(e)
                rec["raw"] = content[:4000]
            total += float(usage.get("cost") or 0)
            with out.open("a") as f:
                f.write(json.dumps(rec) + "\n")
            print(f"{iid} run {run}: {len(rec['spans'])} spans, {len(rec['unmatched'])} unmatched"
                  f"{', ERROR ' + rec['error'] if rec['error'] else ''}  cost so far ${total:.4f}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
