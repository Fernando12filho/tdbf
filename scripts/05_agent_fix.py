"""Phase B: a single autonomous fix agent.

Takes a *validated* Bug Reproduction Test (a Phase A L4 BRT) and a real agent
(model-driven control) edits the buggy source until the BRT passes — without
breaking existing tests. The model is given the issue + the BRT + gold-free tools
(read_file, grep, glob, edit_file, run_brt, run_tests, revert, submit_fix) and
decides the trajectory itself: locate the bug -> edit -> run_brt -> iterate ->
run existing tests -> submit. Termination is the model's call (terminal submit_fix).

This is Phase B of docs/agent_plan.md, downstream of scripts/04_agent_brt.py.

GOLD-FREE: the agent's tools run only on the buggy checkout. Its success signal is
the BRT (its own reproduction) plus existing repo tests — never the held-out
FAIL_TO_PASS / test_patch. The fix patch (git diff of source edits) is captured
after the agent finishes and scored offline through the SWE-bench harness
(applies the held-out test_patch, runs FAIL_TO_PASS/PASS_TO_PASS -> resolved).
The agent never observes that resolved outcome.

Usage:
  export DOCKER_HOST=unix://${HOME}/.colima/default/docker.sock
  # needs a validated BRT in results.jsonl (run scripts/04_agent_brt.py first)
  python3 scripts/05_agent_fix.py --id astropy__astropy-12907 --model anthropic:claude-sonnet-4-6 -v
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tdbf.agent import Agent, ContainerSession, fix_agent_tools
from tdbf.data.instances import InstanceStore, SWEBenchInstance
from tdbf.harness.runner import (
    PredictionRecord,
    read_report,
    run_harness,
    write_predictions,
)
from tdbf.models.llm import OllamaClient, build_client
from tdbf.results import load_best_brt, save_run

# $/1M tokens (input, output), keyed by substring of the model id. Local = $0.
_PRICING = {"opus": (5.0, 25.0), "sonnet": (3.0, 15.0), "haiku": (1.0, 5.0)}


def _cost_usd(model: str, in_tok: int, out_tok: int) -> float | None:
    for key, (pin, pout) in _PRICING.items():
        if key in model:
            return round(in_tok / 1e6 * pin + out_tok / 1e6 * pout, 4)
    return None


SYSTEM_PROMPT = """You are an autonomous bug-fixing agent. Your single goal is to FIX the bug \
described in a GitHub issue by editing the source code of `{repo}` (checked out at the buggy commit \
`{base_commit}`).

You are given a bug-reproduction test (BRT) that currently FAILS because of the bug. Your fix is \
correct when that test PASSES and you have not broken existing behaviour.

Recommended workflow (you decide the specifics and when you are done):
1. Read the issue and the BRT. Use glob/grep/read_file to locate the buggy code.
2. Call run_brt to see the test fail and understand the failure.
3. Use edit_file to apply a fix to the source.
4. Call run_brt again. Iterate edit -> run_brt until it reports outcome=pass.
5. Use run_tests on the SPECIFIC existing test file(s) closest to your change (e.g. the test module \
for the file you edited) to check you did not break related behaviour. Do NOT run whole test \
directories — they are slow and full of failures unrelated to your change.
6. Once run_brt passes and the targeted tests look consistent, call submit_fix.

Rules:
- Fix the ROOT CAUSE in the source. NEVER edit the test to make it pass — the test is fixed and you \
cannot change it (run_brt always uses the original BRT).
- Keep the change minimal and targeted; do not reformat unrelated code.
- IMPORTANT: existing tests in this repo may already fail for reasons unrelated to you (environment, \
network, pre-existing flakiness). A failure reported by run_tests does NOT by itself mean your fix is \
wrong. Only treat it as a regression if you can point to a specific test that your edit plausibly broke.
- Once run_brt reports outcome=pass you have a working fix. Do NOT call revert on a fix that makes the \
BRT pass unless you have clearly identified a regression YOUR change introduced. When unsure, keep the \
fix and submit. revert discards ALL your edits — use it only to escape a state that is clearly worse.
- Call submit_fix EXACTLY ONCE, only after run_brt reports outcome=pass.
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
    brt_override: str | None,
    keep_container: bool,
    verbose: bool,
    transcript_dir: Path,
) -> bool | None:
    print(f"\n{'='*60}\nInstance:  {inst.instance_id}\nRepo:      {inst.repo}")

    # --- Obtain the validated BRT the fix must make pass. ---
    if brt_override is not None:
        brt_code = Path(brt_override).read_text()
        brt_src = f"file:{brt_override}"
    else:
        brt = load_best_brt(inst.instance_id, min_level=4)
        if brt is None:
            print("  [skip] No validated (L4) BRT for this instance in results.jsonl. "
                  "Run scripts/04_agent_brt.py first or pass --brt-file.")
            return None
        brt_code = brt["test_code"]
        brt_src = f"results.jsonl(L{brt['level']}, {brt['model']})"
    print(f"BRT:       {brt_src}")

    client = build_client(model_spec)
    if think and isinstance(client, OllamaClient):
        client._think = True
    print(f"Model:     {client.name}")

    system = SYSTEM_PROMPT.format(repo=inst.repo, base_commit=inst.base_commit)
    ctx = inst.context_safe_dict()
    task = (
        f"GitHub issue for `{ctx['instance_id']}` ({ctx['repo']} @ {ctx['base_commit']}):\n\n"
        f"{ctx['problem_statement']}\n\n"
        f"--- Bug-reproduction test (currently FAILS; make it PASS by fixing the source) ---\n"
        f"{brt_code}"
    )

    # --- AGENT (gold-free): the model edits the buggy tree. ---
    try:
        session = ContainerSession(inst, keep=keep_container).start()
    except RuntimeError as e:
        print(f"  [skip] {e}")
        return None
    try:
        tools = fix_agent_tools(session, brt_code)  # gold-free; BRT baked into run_brt
        agent = Agent(
            client, tools, system,
            max_iterations=max_iterations,
            max_total_tokens=max_total_tokens,
            verbose=verbose,
        )
        print(f"Running fix agent (max_iter={max_iterations}, max_tok={max_total_tokens}) ...")
        result = agent.run(task)
        # Capture the fix patch = tracked source edits (the untracked BRT is excluded).
        fix_patch = session.exec("git diff").output
    finally:
        session.close()

    cost = _cost_usd(client.name, result.total_input_tokens, result.total_output_tokens)
    print(
        f"Agent done: stop={result.stop_cause}  iters={result.iterations}  "
        f"tokens={result.total_input_tokens}/{result.total_output_tokens}"
        + (f"  cost=${cost}" if cost is not None else "  cost=$0 (local)")
        + f"  patch={len(fix_patch)} chars"
    )

    # =================== GOLD BARRIER =================== #
    # Below uses the held-out test_patch (via the SWE-bench harness) PURELY to
    # decide resolved/not. The agent never saw any of this.
    resolved = False
    harness = None
    if not fix_patch.strip():
        print("  Agent produced no source edits — not resolved.")
    else:
        run_id = f"agent_fix_{inst.instance_id}"
        pred = PredictionRecord(
            instance_id=inst.instance_id,
            model_name_or_path=client.name,
            model_patch=fix_patch,
        )
        pred_path = write_predictions([pred], Path("runs") / "agent_fix" / f"{run_id}.jsonl")
        print("Scoring offline via SWE-bench harness (held-out test_patch) ...")
        run_harness(pred_path, split, run_id, [inst.instance_id])
        harness = read_report(run_id, client.name, inst)
        resolved = harness.resolved
        print(f"  applied={harness.applied}  resolved={resolved}  "
              f"FAIL_TO_PASS pass/fail={len(harness.fail_to_pass_passed)}/"
              f"{len(harness.fail_to_pass_failed)}  PASS_TO_PASS fail={len(harness.pass_to_pass_failed)}")

    # --- Persist transcript + summary record. ---
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
                "fix_patch": fix_patch,
                "resolved": resolved,
            },
            indent=2,
        )
    )

    attempt_record = {
        "attempt": 1,
        "mode": "fix_agent",
        "resolved": resolved,
        "applied": harness.applied if harness else False,
        "stop_cause": result.stop_cause,
        "iterations": result.iterations,
        "num_tool_calls": sum(len(s.tool_calls) for s in result.steps),
        "fix_patch": fix_patch,
        "fail_to_pass_failed": harness.fail_to_pass_failed if harness else None,
        "pass_to_pass_failed": harness.pass_to_pass_failed if harness else None,
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
        "mode": "fix_agent",
        "flags": {"max_iterations": max_iterations, "max_total_tokens": max_total_tokens},
        "best_level": 4 if resolved else 0,  # reuse field: resolved -> sentinel
        "resolved": resolved,
        "best_attempt": 1,
        "attempts": [attempt_record],
    })
    return resolved


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", default="lite", choices=["lite", "verified"])
    ap.add_argument("--id", default=None, help="instance_id (default: first in split)")
    ap.add_argument("--limit", type=int, default=None,
                    help="Run the first N instances of the split (ignored if --id given).")
    ap.add_argument("--model", default="anthropic:claude-sonnet-4-6",
                    help="Model spec: anthropic:<model>, ollama:<model>@<url>, openai:<model>")
    ap.add_argument("--brt-file", default=None,
                    help="Path to a BRT test file to use instead of looking one up in results.jsonl.")
    ap.add_argument("--max-iterations", type=int, default=20)
    ap.add_argument("--max-tokens", type=int, default=300_000,
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

    transcript_dir = Path("runs") / "agent_fix"
    results: list[bool] = []
    for inst in instances:
        r = run_one(
            inst, args.model,
            split=args.split,
            max_iterations=args.max_iterations,
            max_total_tokens=args.max_tokens,
            think=args.think,
            brt_override=args.brt_file,
            keep_container=args.keep_container,
            verbose=args.verbose,
            transcript_dir=transcript_dir,
        )
        if r is not None:
            results.append(r)

    if len(results) > 1:
        n_resolved = sum(results)
        print(f"\n{'='*60}\nFix agent on {len(results)} instance(s): "
              f"resolved {n_resolved}/{len(results)} = {n_resolved/len(results):.2f}")

    return 0 if any(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
