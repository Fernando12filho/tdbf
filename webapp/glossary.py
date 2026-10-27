"""Plain-language definitions for every term the Results and Development pages use.

One source for both pages: a template links a word to its entry with the `term`
macro, and the glossary section at the bottom of each page renders from here.
Written for a reader outside the project, so nothing is assumed, not even
"sweep" or "instance". Metrics carry their formula.
"""

from __future__ import annotations

# (section title, [(key, term, definition, formula or None), ...])
SECTIONS: list[tuple[str, list[tuple[str, str, str, str | None]]]] = [
    ("The benchmark", [
        ("swebench", "SWE-bench Lite",
         "A public benchmark of 300 real bugs taken from 12 open-source Python "
         "projects. Each bug comes with hidden tests that check whether a fix works.",
         None),
        ("instance", "Instance",
         "One bug from SWE-bench Lite: the GitHub issue describing it, the project's "
         "code at the commit where the bug exists, and the hidden tests.",
         None),
        ("repo", "Repository (repo)",
         "The open-source project an instance comes from, for example django or sympy.",
         None),
        ("frozen_sample", "Frozen sample",
         "The 60 instances chosen for the experiment. They were picked at random, "
         "stratified by repository (each project gets a share matching its share of "
         "SWE-bench Lite), and saved to a file before any run, so the results could "
         "not influence which bugs were included.",
         None),
        ("docker_image", "Docker image",
         "A packaged copy of a project with the exact Python version and libraries "
         "the bug needs, so tests behave the same way on every run.",
         None),
        ("left_out", "Incomplete / no image",
         "Why an instance from the frozen sample has no result. Incomplete: the run "
         "was attempted, but the model provider kept returning errors (overloaded), "
         "so it never finished. No image: its Docker image was not available, so it "
         "was never run.",
         None),
    ]),
    ("How the experiment was run", [
        ("rq", "Research questions (RQ1, RQ3)",
         "The questions the project sets out to answer. RQ1: does writing a test "
         "before fixing resolve more bugs than going straight to fixing? RQ3: can a "
         "free, open-weight model carry the whole process, and at which step does it "
         "break down? (RQ2, on which parts of the setup help, is answered from the "
         "development runs.)",
         None),
        ("llm", "Model (LLM)",
         "The AI language model that does the work. Here: nvidia "
         "nemotron-3-super-120b, used through OpenRouter's free tier.",
         None),
        ("open_weight", "Open-weight model",
         "A model whose trained weights are published, so anyone can download and "
         "run it, unlike closed models that are only reachable through a company's "
         "service.",
         None),
        ("openrouter", "OpenRouter",
         "A service that gives access to many language models through one interface. "
         "Its free tier allows about 1,000 model calls per day.",
         None),
        ("agent", "Agent",
         "A language model working in a loop: at each step it chooses an action "
         "(read a file, search the code, run a test, edit a file) through tools, "
         "sees the result, and decides what to do next, until it decides it is done "
         "or runs out of steps.",
         None),
        ("iteration", "Iteration (turn)",
         "One step of an agent's loop: the model answers once, usually calling one "
         "tool. Each phase of an agent is limited to 30 iterations.",
         None),
        ("model_call", "Model call",
         "One request to the language model. The free tier limits calls per day, so "
         "calls are how we measure cost.",
         None),
        ("sweep", "Sweep",
         "One automated run of an agent setup over a whole list of instances, one "
         "after another. The main sweep ran the test-first setup over the frozen "
         "sample; a second sweep ran the reactive setup over the same instances.",
         None),
        ("arm", "Arm",
         "One of the setups being compared. This experiment has two: test-first "
         "and reactive.",
         None),
        ("test_first", "Test-first arm",
         "An agent first writes a test that reproduces the bug (a BRT). Only if that "
         "test is confirmed to fail on the buggy code does a second agent fix the "
         "code, and it can run that test to check its progress.",
         None),
        ("reactive", "Reactive arm (baseline)",
         "The same fixing agent, with the same model and limits, but no test is "
         "written first: it goes straight to fixing the code. It is the point of "
         "comparison for the test-first arm.",
         None),
        ("paired", "Paired",
         "Every instance was attempted by both arms, so results are compared bug by "
         "bug rather than as two separate groups.",
         None),
        ("stopping", "Stopped itself / ran out of turns",
         "How a fixing agent's run ended. Stopped itself: it declared the fix done. "
         "Ran out of turns: it reached the 30-iteration limit and the system made it "
         "hand in whatever it had.",
         None),
    ]),
    ("Code, tests and fixes", [
        ("buggy_code", "Buggy code",
         "The project's code at the commit where the bug exists.",
         None),
        ("gold_patch", "Gold patch",
         "The real fix written by the project's developers. It is used only "
         "afterwards, to score results. The agents never see it.",
         None),
        ("fixed_code", "Fixed code",
         "The buggy code with the gold patch applied.",
         None),
        ("gold_free", "Gold-free",
         "Never uses the gold patch or the hidden tests. Everything the agents do, "
         "including the gate, is gold-free.",
         None),
        ("patch", "Patch",
         "A set of changes to the code, written as a diff. The agent's fix is a patch.",
         None),
        ("hidden_tests", "Hidden tests",
         "Tests that SWE-bench keeps aside to judge a fix. The agents never see them. "
         "They come in two kinds: F2P and P2P.",
         None),
        ("f2p", "F2P (FAIL_TO_PASS)",
         "Hidden tests that fail on the buggy code and must pass after a correct fix. "
         "They check that the bug itself is fixed.",
         None),
        ("p2p", "P2P (PASS_TO_PASS)",
         "Hidden tests that already pass on the buggy code and must still pass after "
         "the fix. They check that the fix did not break anything else.",
         None),
        ("resolved", "Resolved",
         "The bug counts as fixed. Decided by SWE-bench's official scorer.",
         "resolved = every F2P test passes AND every P2P test still passes"),
        ("regression", "Regression",
         "A fix that breaks something that used to work, i.e. a P2P test fails.",
         None),
        ("verification_gap", "Verification gap",
         "The fix makes the agent's own test pass, but the hidden tests still fail. "
         "The agent had no way to see the difference.",
         None),
    ]),
    ("The reproduction test and the gate", [
        ("brt", "BRT (bug reproduction test)",
         "A test written by the agent that should fail on the buggy code because of "
         "the bug, and pass once the bug is fixed.",
         None),
        ("gate", "Gate",
         "The check between writing the test and fixing the code. The BRT is "
         "accepted only if, on the buggy code, it runs, reaches its check and fails "
         "there (rather than crashing earlier). If it is rejected, no fix is "
         "attempted. The gate never looks at the fixed code, so it cannot tell L3 "
         "from L4.",
         None),
        ("gate_retry", "Gate retry",
         "When the gate rejects a test, the test-writing agent gets up to 2 more "
         "tries, with feedback on what went wrong.",
         None),
        ("levels", "BRT level (L0 to L4)",
         "How a test behaves on the buggy code and on the fixed code. Measured "
         "afterwards, using the gold patch. "
         "L0: no usable test was produced. "
         "L1: passes on the buggy code, so it does not reproduce the bug. "
         "L2: errors on the buggy code (crash, import error) before checking anything. "
         "L3: fails on the buggy code but also fails on the fixed code; it catches "
         "a failure, but checks the wrong thing. "
         "L4: fails on the buggy code and passes on the fixed code; a valid BRT.",
         None),
    ]),
    ("Metrics", [
        ("n", "N",
         "The number of instances the metric is computed over. On the Results page, "
         "N is the number of instances attempted by both arms.",
         None),
        ("c2p", "C2P",
         "Of the tests that fail on the buggy code, the share that are valid. It "
         "answers: when the agent's test fails, is it failing for the right reason?",
         "C2P = L4 / (L3 + L4)"),
        ("valid_brt", "Valid-BRT",
         "The share of all instances that ended up with a valid test.",
         "Valid-BRT = L4 / N"),
        ("gate_rate", "GateRate",
         "The share of instances whose test passed the gate, so a fix was attempted.",
         "GateRate = passed the gate / N"),
        ("resolve_at_gated", "Resolve@Gated",
         "Of the instances that passed the gate, the share that were resolved.",
         "Resolve@Gated = resolved / passed the gate"),
        ("resolve_rate", "Resolved rate (Pipeline resolve)",
         "The share of all instances that were resolved. The main outcome measure "
         "for both arms.",
         "Resolved rate = resolved / N  =  GateRate x Resolve@Gated (test-first)"),
        ("calls_per_instance", "Calls per instance",
         "The average number of model calls one instance cost, counting every phase "
         "of the arm.",
         "Calls per instance = total model calls / N"),
    ]),
    ("SWT-bench re-scoring", [
        ("swt_bench", "SWT-bench Lite",
         "A public benchmark for writing tests that reproduce a bug: 276 of the "
         "SWE-bench Lite bugs, with an official scorer. Other tools that write such "
         "tests (SWE-agent, LIBRO, Issue2Test) report their results on it.",
         None),
        ("rescoring", "Re-scoring",
         "Taking the tests our agent already wrote, with no new model calls, and "
         "judging them a second time with SWT-bench's official scorer instead of ours, "
         "so our numbers can sit next to other tools' numbers.",
         None),
        ("swt_harness", "SWT-bench harness",
         "SWT-bench's official scorer. It adds the test to the project, runs it with "
         "the project's own test command on the buggy code and on the fixed code, and "
         "records what each test function did.",
         None),
        ("swt_success", "S (success rate)",
         "The strict SWT-bench measure. A test counts only if at least one test "
         "function goes from failing to passing and none still fails after the fix. "
         "Our L4 is meant to match this.",
         "S = instances with ≥1 fail→pass test and no test failing after the fix / N"),
        ("swt_f2p", "F→P (fail-to-pass rate)",
         "The loose SWT-bench measure. A test counts if at least one test function "
         "goes from failing to passing, even if another still fails after the fix. "
         "Always at least S. The figure Issue2Test's paper reports.",
         "F→P = instances with ≥1 fail→pass test / N"),
        ("swt_applied", "W (applicability)",
         "The share of tests the scorer could add to the project without an error.",
         "W = tests that applied / N"),
        ("coverage_delta", "Coverage delta",
         "How many of the lines the real fix changes the test runs, beyond what the "
         "project's existing tests already run. Higher means the test exercises the "
         "code the bug is in.",
         None),
        ("agreement", "Agreement",
         "The share of instances where SWT-bench and our scorer give the same answer "
         "to: is this a valid test? (Our L4 against SWT-bench's S.)",
         "Agreement = (L4 and S, or not L4 and not S) / N"),
        ("published_lists", "Published results",
         "Which instances other tools solved, as published by their authors. They "
         "used other models (mostly GPT-4, from 2023), so these rows compare tools "
         "and models together, not tools alone.",
         None),
    ]),
    ("Development terms", [
        ("dev_set", "Development set",
         "The 14 instances (6 astropy, 8 django) used while building the agents. "
         "They are excluded from the frozen sample.",
         None),
        ("run_group", "Run group",
         "A tag on every result naming the run it came from, so results from "
         "different runs are never mixed in one number.",
         None),
        ("probe", "Probe",
         "A small trial run used to check that the setup works before a sweep.",
         None),
        ("ablation", "Ablation",
         "An experiment that adds or removes one component to see what it changes.",
         None),
        ("roles", "Critic, Reviewer, Breaker, Selector, best-of-N",
         "Extra agent roles tried during development. Critic: reviews a test before "
         "it is accepted. Reviewer: reviews a fix. Breaker: tries to find a case the "
         "fix still gets wrong. Selector with best-of-N: writes several tests and "
         "picks one.",
         None),
        ("resolve_at_brt", "Resolve@BRT",
         "In development fix tables: of the instances that reached the fixing "
         "phase with a test, the share that were resolved.",
         "Resolve@BRT = resolved / instances that reached the fix phase"),
        ("ceiling", "L4-gated ceiling",
         "An early pipeline whose gate accepted only L4 tests. Telling L4 from L3 "
         "needs the gold patch, so a real deployment could not do this; it is an "
         "upper bound, not a usable setup.",
         None),
        ("fix_only", "Fix-only",
         "The fixing agent given a test validated in an earlier run, without "
         "running the test-writing phase in the same run.",
         None),
        ("local", "Local",
         "A model running on our own machine (through Ollama) instead of a hosted "
         "service. It costs nothing per call.",
         None),
        ("non_agentic", "Non-agentic baseline",
         "An older fixed loop (scripts 02/03): the program, not the model, decides "
         "the steps: ask for a test, run it, retry. Its retry stops at the first L4, "
         "which needs the gold patch, so its numbers are not comparable with the "
         "gold-free agents.",
         None),
        ("cost_usd", "Cost ($)",
         "Money paid for paid models. Free and local models cost $0, which is why "
         "the Results page counts model calls instead.",
         None),
    ]),
]

TERMS: dict[str, dict] = {
    key: {"term": term, "definition": definition, "formula": formula}
    for _, entries in SECTIONS
    for key, term, definition, formula in entries
}

# Which sections each page shows at the bottom.
RESULTS_SECTIONS = [s for s in SECTIONS if s[0] not in ("Development terms", "SWT-bench re-scoring")]
DEVELOPMENT_SECTIONS = [s for s in SECTIONS if s[0] != "SWT-bench re-scoring"]

# The Status page uses only a few terms; show just those, in glossary order.
STATUS_TERMS = ("swebench", "instance", "repo", "frozen_sample", "docker_image",
                "arm", "paired", "gold_patch", "dev_set", "run_group")


SWT_TERMS = ("instance", "repo", "docker_image", "buggy_code", "gold_patch", "fixed_code",
             "llm", "brt", "levels", "n", "valid_brt",
             "swt_bench", "rescoring", "swt_harness", "swt_success", "swt_f2p",
             "swt_applied", "coverage_delta", "agreement", "published_lists")


def sections_for(keys) -> list:
    """SECTIONS narrowed to the given term keys (empty sections dropped)."""
    wanted = set(keys)
    out = []
    for title, entries in SECTIONS:
        kept = [e for e in entries if e[0] in wanted]
        if kept:
            out.append((title, kept))
    return out
