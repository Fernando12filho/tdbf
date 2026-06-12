"""Phase A: a single autonomous BRT agent.

A real agent (model-driven control) writes a Bug Reproduction Test for a
SWE-bench instance. The model is given the issue + gold-free tools (read_file,
grep, glob, run_python, run_test-on-buggy, submit_test) and decides the
trajectory itself: explore -> observe buggy behaviour -> write test -> run it ->
iterate -> submit. Termination is the model's call (the terminal submit_test
tool), not a fixed loop.

This is the agentic counterpart to scripts/02_generate_brt.py (the workflow
baseline). The same Agent loop runs on Claude or qwen (RQ1) by changing --model.

GOLD-FREE: the agent's tools run only on the buggy checkout. The gold patch is
applied ONLY in the offline scoring section below, after the agent has finished,
to assign an L0-L4 level. The agent never observes the fixed-tree outcome.

Usage:
  export DOCKER_HOST=unix://${HOME}/.colima/default/docker.sock
  python3 scripts/01_validate_gold.py --split lite --id astropy__astropy-12907   # build image
  python3 scripts/04_agent_brt.py --id astropy__astropy-12907 --model anthropic:claude-opus-4-8 -v
  python3 scripts/04_agent_brt.py --id astropy__astropy-12907 \\
      --model ollama:qwen2.5-coder@http://192.168.1.192:11434       # weak baseline
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tdbf.agent import Agent, ContainerSession, brt_agent_tools
from tdbf.data.instances import InstanceStore, SWEBenchInstance
from tdbf.harness.runner import (
    brt_test_path,
    inject_setup_preamble,
    make_test_patch,
    repo_test_hint,
    run_test_in_container,
)
from tdbf.models.llm import OllamaClient, build_client
from tdbf.results import save_run
from tdbf.validation.brt_levels import BRTClassification, classify, c2p_rate

# $/1M tokens (input, output). Keyed by substring of the model id. See
# docs/cost_estimate.md. Unknown models (ollama/local) cost $0.
_PRICING = {
    "opus": (5.0, 25.0),
    "sonnet": (3.0, 15.0),
    "haiku": (1.0, 5.0),
}


def _cost_usd(model: str, in_tok: int, out_tok: int) -> float | None:
    for key, (pin, pout) in _PRICING.items():
        if key in model:
            return round(in_tok / 1e6 * pin + out_tok / 1e6 * pout, 4)
    return None  # local / unknown


SYSTEM_PROMPT = """You are an autonomous software-testing agent. Your single goal is to write a \
Bug Reproduction Test (BRT): one pytest test that reproduces the bug described in a GitHub issue.

You are working inside a checkout of `{repo}` at the BUGGY commit `{base_commit}`. The bug has NOT \
been fixed. Your tools run against this buggy code.

A good BRT:
- FAILS on this buggy code with an ASSERTION failure that directly reflects the bug in the issue.
- Fails for the RIGHT reason: it must reach its assertion and fail there, NOT crash with an \
ImportError / NameError / setup error before exercising the behaviour.
- Would PASS once the bug is correctly fixed. You cannot run the fixed code, so reason from the \
issue about what the correct behaviour should be and assert exactly that.

Recommended workflow (you decide the specifics and when you are done):
1. Read the issue. Use glob/grep/read_file to locate the relevant code.
2. Use run_python to OBSERVE the actual buggy behaviour and confirm your understanding before \
asserting anything.
3. Write the test and use run_test to check it FAILS on the buggy code with failure_reason=assertion.
4. Iterate until run_test reports outcome=fail with failure_reason=assertion, then call submit_test.

Rules:
- Framework setup boilerplate is injected for you automatically. Write plain pytest functions; do \
NOT write framework bootstrap (no settings.configure(), no django.setup()).
- {env_hint}
- The test file is placed for you; do not worry about its path or filename.
- Only assert behaviour you are confident a correct fix would produce.
- Call submit_test EXACTLY ONCE, only after run_test has confirmed a clean assertion failure on the \
buggy code. Never submit a test that passes or merely errors on the buggy code.
- Be efficient — you have a limited number of steps and a token budget."""


def _serialise_messages(messages) -> list[dict]:
    out = []
    for m in messages:
        d = {"role": m.role}
        if m.content is not None:
            d["content"] = m.content
        if m.tool_calls:
            d["tool_calls"] = [
                {"id": tc.id, "name": tc.name, "arguments": tc.arguments} for tc in m.tool_calls
            ]
        if m.tool_call_id:
            d["tool_call_id"] = m.tool_call_id
        out.append(d)
    return out


def _step_summary(steps) -> list[dict]:
    return [
        {
            "iteration": s.iteration,
            "tools": [tc.name for tc in s.tool_calls],
            "input_tokens": s.input_tokens,
            "output_tokens": s.output_tokens,
            "duration_seconds": s.duration_seconds,
        }
        for s in steps
    ]


def run_one(
    inst: SWEBenchInstance,
    model_spec: str,
    *,
    split: str,
    max_iterations: int,
    max_total_tokens: int,
    think: bool,
    keep_container: bool,
    verbose: bool,
    transcript_dir: Path,
) -> BRTClassification | None:
    print(f"\n{'='*60}\nInstance:  {inst.instance_id}\nRepo:      {inst.repo}")

    client = build_client(model_spec)
    if think and isinstance(client, OllamaClient):
        client._think = True
    print(f"Model:     {client.name}")

    test_path = brt_test_path(inst.repo)
    env_hint = repo_test_hint(inst.repo) or "(no special framework setup for this repo)"
    system = SYSTEM_PROMPT.format(
        repo=inst.repo, base_commit=inst.base_commit, env_hint=env_hint
    )
    ctx = inst.context_safe_dict()
    task = (
        f"GitHub issue for instance `{ctx['instance_id']}` "
        f"({ctx['repo']} @ {ctx['base_commit']}):\n\n{ctx['problem_statement']}"
    )

    # --- AGENT (gold-free): the model drives the tools on the buggy tree. ---
    try:
        session = ContainerSession(inst, keep=keep_container).start()
    except RuntimeError as e:
        print(f"  [skip] {e}")
        return None
    try:
        tools = brt_agent_tools(session)  # gold-free set + assert_gold_free guard
        agent = Agent(
            client, tools, system,
            max_iterations=max_iterations,
            max_total_tokens=max_total_tokens,
            verbose=verbose,
        )
        print(f"Running agent (max_iter={max_iterations}, max_tok={max_total_tokens}) ...")
        result = agent.run(task)
    finally:
        session.close()

    cost = _cost_usd(client.name, result.total_input_tokens, result.total_output_tokens)
    print(
        f"Agent done: stop={result.stop_cause}  iters={result.iterations}  "
        f"tokens={result.total_input_tokens}/{result.total_output_tokens}"
        + (f"  cost=${cost}" if cost is not None else "  cost=$0 (local)")
    )

    # =================== GOLD BARRIER =================== #
    # Everything above is gold-free. Below, the gold patch is used PURELY to
    # score the finished test into L0-L4. The agent never saw any of this.
    submission = result.submission
    classification = None
    outcome_buggy = outcome_fixed = None
    if not submission or not submission.strip():
        print("  Agent did not submit a test (stop_cause="
              f"{result.stop_cause}). Scoring as L0.")
    else:
        test_code = inject_setup_preamble(inst.repo, submission)
        test_patch = make_test_patch(test_path, test_code)
        print("Scoring offline — BUGGY tree ...")
        outcome_buggy = run_test_in_container(inst, test_patch, test_path=test_path)
        print(f"  outcome_on_buggy:  {outcome_buggy}")
        print("Scoring offline — gold-FIXED tree ...")
        outcome_fixed = run_test_in_container(
            inst, test_patch, gold_patch=inst.gold_patch, test_path=test_path
        )
        print(f"  outcome_on_fixed:  {outcome_fixed}")
        classification = classify(outcome_buggy, outcome_fixed)
        print(f"  -> {classification.level_label}  (valid BRT: {classification.is_valid_brt})")

    # --- Persist full transcript + summary record. ---
    transcript_dir.mkdir(parents=True, exist_ok=True)
    (transcript_dir / f"{inst.instance_id}.json").write_text(
        json.dumps(
            {
                "instance_id": inst.instance_id,
                "model": client.name,
                "system": system,
                "task": task,
                "stop_cause": result.stop_cause,
                "messages": _serialise_messages(result.messages),
                "submission": submission,
            },
            indent=2,
        )
    )

    level = classification.level.value if classification else 0
    attempt_record = {
        "attempt": 1,
        "mode": "agent",
        "level": level,
        "l3_subtype": classification.l3_subtype if classification else None,
        "stop_cause": result.stop_cause,
        "iterations": result.iterations,
        "num_tool_calls": sum(len(s.tool_calls) for s in result.steps),
        "outcome_buggy": str(outcome_buggy) if outcome_buggy is not None else None,
        "outcome_fixed": str(outcome_fixed) if outcome_fixed is not None else None,
        "test_code": inject_setup_preamble(inst.repo, submission) if submission else None,
        "input_tokens": result.total_input_tokens,
        "output_tokens": result.total_output_tokens,
        "cost_usd": cost,
        "duration_seconds": sum((s.duration_seconds or 0) for s in result.steps),
        "step_summary": _step_summary(result.steps),
    }
    save_run({
        "instance_id": inst.instance_id,
        "repo": inst.repo,
        "model": client.name,
        "split": split,
        "mode": "agent",
        "flags": {"max_iterations": max_iterations, "max_total_tokens": max_total_tokens},
        "best_level": level,
        "best_attempt": 1,
        "attempts": [attempt_record],
    })
    return classification


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", default="lite", choices=["lite", "verified"])
    ap.add_argument("--id", default=None, help="instance_id (default: first in split)")
    ap.add_argument("--limit", type=int, default=None,
                    help="Run the first N instances of the split (ignored if --id given).")
    ap.add_argument("--model", default="anthropic:claude-opus-4-8",
                    help="Model spec: anthropic:<model>, ollama:<model>@<url>, openai:<model>")
    ap.add_argument("--max-iterations", type=int, default=20,
                    help="Iteration cap. 15 was too low for harder instances "
                         "(astropy-6938 timed out mid-exploration → spurious L0; "
                         "submitted L4 at iter 13 with cap 25).")
    ap.add_argument("--max-tokens", type=int, default=200_000,
                    help="Cumulative token budget for one agent run (hard cap).")
    ap.add_argument("--think", action="store_true", help="Enable Qwen3 thinking mode (Ollama).")
    ap.add_argument("--keep-container", action="store_true",
                    help="Do not remove the agent's container after the run (debugging).")
    ap.add_argument("-v", "--verbose", action="store_true", help="Stream the agent's tool trace.")
    args = ap.parse_args()

    store = InstanceStore(args.split)
    if args.id:
        instances = [store.get(args.id)]
    elif args.limit:
        instances = list(store.iter_instances(limit=args.limit))
    else:
        instances = [store.first()]

    transcript_dir = Path("runs") / "agent_brt"
    classifications: list[BRTClassification] = []
    for inst in instances:
        c = run_one(
            inst, args.model,
            split=args.split,
            max_iterations=args.max_iterations,
            max_total_tokens=args.max_tokens,
            think=args.think,
            keep_container=args.keep_container,
            verbose=args.verbose,
            transcript_dir=transcript_dir,
        )
        if c is not None:
            classifications.append(c)

    # --- Aggregate report ---
    if len(instances) > 1:
        print(f"\n{'='*60}\nBatch of {len(instances)} instance(s):")
        dist = {}
        for c in classifications:
            dist[c.level.name] = dist.get(c.level.name, 0) + 1
        for name in sorted(dist):
            print(f"  {name}: {dist[name]}")
        rate = c2p_rate(classifications)
        print(f"C2P rate (L4 / (L3+L4)): {rate if rate is not None else 'n/a (no candidates)'}")

    any_valid = any(c.is_valid_brt for c in classifications)
    return 0 if any_valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
