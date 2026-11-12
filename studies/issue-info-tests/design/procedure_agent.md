# Design: the procedure agent for bug reproduction tests

**Status: DESIGN DRAFT, 2026-11-12.** Input for Phase 1 of `../plan.md` (Amendment 2). Nothing
here is built yet. This document is meant to be handed to the agent that implements it.

## 0. What changes, in one paragraph

Today the test agent is one loop: one system prompt, all tools available at every step, 30
iterations, and up to two cold retries with fixed feedback ("it passed, make it fail"). The
procedure agent puts a **controller written in Python** around the model. The controller holds
the current phase, decides which tools the model may call, checks each phase's result with
**deterministic code (no model call)**, and starts each phase with a **short handoff record**
instead of the whole transcript. A failed check sends the run back to the phase that caused the
failure, not to the beginning. Every expected value the test asserts must carry evidence that the
controller can verify mechanically, or be flagged as an unknown.

### Principles (each one is a requirement)

1. **Gold-free.** No phase sees the fixed code, the gold patch or the hidden tests. The gold patch
   is used only offline for the L0–L4 label and the SWT-bench verdict.
2. **The controller decides, the model proposes.** Phase order, tool permissions, budgets and gate
   decisions live in Python. The model only works inside a phase and declares its result through
   that phase's `finish_*` tool.
3. **Gates are deterministic.** String matching, parsing test output, comparing values. If a gate
   needs judgement it is not a gate; it becomes a flag in the record.
4. **Same total budget as today.** Phase budgets sum to 30 model turns per attempt, so any gain
   comes from structure, not from more steps.
5. **Nothing lost.** Every phase transcript, gate result and handoff record is stored, including
   attempts that were sent back.

## 1. How it is set up (pseudocode)

```python
# src/tdbf/agent/procedure.py  (new)

PHASES = ["preflight", "understand", "reproduce", "pin_expected", "locate", "write", "finalise"]

@dataclass
class Evidence:                    # one claim the test relies on
    kind: str                      # "observed" | "expected"
    value: str                     # e.g. "models.Outer.Inner", "raises ValueError"
    source: str                    # "issue" | "docs" | "existing_test" | "observed_run" | "analogy" | "model"
    quote: str | None              # verbatim text backing it (issue, file content, or run output)
    where: str | None              # file path for docs/existing_test, run id for observed_run
    verified: bool = False         # set by the controller, never by the model

@dataclass
class Record:                      # the handoff carried between phases
    instance_id: str
    issue_text: str
    env: EnvProfile                # from preflight: test command, framework, test dirs
    issue_snippets: list[Snippet]  # code blocks from the issue, with preflight run results
    candidates: Candidates         # code and test-file candidates found in preflight
    claims: list[Evidence]         # observed + expected claims
    unknowns: list[Unknown]        # expected values with no verified evidence yet
    repro: Repro | None            # script, output, symptom_matched
    test_home: TestHome | None     # file path / Django module, runner label
    test: str | None
    run_on_buggy: TestRun | None   # outcome, failure_reason, failing assertion, message
    flags: list[str]               # "unreproduced", "unknown_expected", "symptom_partial", ...
    history: list[GateResult]      # every gate decision, including failures

def run_procedure(inst, client, budget=PHASE_BUDGET, mode="strict"):
    session = ContainerSession(inst).start()           # SWE-bench prebuilt image, as today
    rec = Record(inst.instance_id, inst.problem_statement, ...)
    rec = preflight(session, inst, rec)                # no model call
    if rec.env.invalid:
        return finish(rec, outcome="environment_invalid")   # excluded, as script 13 does today

    phase, turns_left = "understand", TOTAL_TURNS      # 30
    while phase != "done" and turns_left > 0:
        spec = PHASE_SPECS[phase]
        tools = gated_tools(session, spec.allowed_tools, rec) + [spec.finish_tool]
        prompt = build_prompt(spec, rec)               # system + phase task + handoff (not history)
        result = run_agent_loop(client, prompt, tools,
                                max_iterations=min(spec.budget, turns_left),
                                stop_on=spec.finish_tool.name)
        turns_left -= result.iterations
        rec.history.append(store_transcript(phase, result))

        gate = spec.gate(session, rec, result.finish_args)    # deterministic check
        rec.history.append(gate)
        if gate.passed:
            rec = spec.apply(rec, result.finish_args, gate)   # write verified fields into the record
            phase = spec.next_phase(rec)                      # may skip phases, see §3
        else:
            phase, note = spec.on_fail(gate, rec)             # back to the phase that can fix it
            rec.flags.append(note)
            if spec.attempts(phase) > MAX_VISITS[phase]:      # stop loops
                phase = degrade(rec, phase)                   # e.g. continue flagged, see §5

    rec = finalise(rec)                                       # flags, evidence summary, submission
    session.close()
    return rec

# "soft" mode for the comparison: same prompts and phases as one prompt, all tools at all times,
# no gates. Lets Phase 1 separate the effect of the steps from the effect of enforcing them.
```

`run_agent_loop` is the existing `AgentRuntime` loop (`src/tdbf/agent/runtime.py`); only the tools,
prompt and stop condition change per phase. The existing tools (`read_file`, `grep`, `glob`,
`run_python`, `run_test`) are reused; new tools are listed per phase.

## 2. Environment: running any SWE-bench repository reliably

| Concern | Today | Procedure agent |
|---|---|---|
| Image | SWE-bench prebuilt `sweb.eval.x86_64.<id>`, one container per run, conda env `testbed` (`session.py`) | Same. Reuse the image across all repeats and conditions of a bug (change in `scripts/16_nightly_batch.py`) |
| How a test runs | pytest everywhere, except Django via `runtests.py`; sympy adds `-o python_functions=test_*` (`runner.py`) | Use the **official command from SWE-bench**: `MAP_REPO_VERSION_TO_SPECS[repo][version]["test_cmd"]` (swebench 4.1.0): e.g. sympy `bin/test -C --verbose`, sphinx `tox --current-env -epy39 -v --`, Django `runtests.py`, others `pytest -rA`. This is how the grader runs tests, so our verdict and SWT-bench's cannot disagree on mechanics |
| Where the test file lives | repo root `test_brt.py` (non-Django); `tests/<module>/test_brt.py` (Django); SWT export moves sympy to `sympy/core/tests` | In the test directory chosen in the *locate* phase (§3.5), so conftest files, fixtures and the official runner apply |
| Reading results | our parsers (`parse_brt_result`, `parse_runtests_result`) | Keep them for outcome and failure reason; add SWE-bench's per-repo log parser (`swebench.harness.log_parsers`) to read the status of the named test |
| Broken environments | script 13 checks, before any run, that the gold patch resolves and a trivial failing/passing test behaves; failures are excluded | Same check, run in *preflight*, plus: the adapter's own trivial test must run with the official command |
| Timeouts | enforced inside the container (`timeout -s KILL`), 180 s for tests, 90 s for probes | Same |

**Preflight checks before any model call (`EnvProfile`):**
1. Start the container; `python -c "import <top-level package>"` succeeds.
2. A trivial passing and a trivial failing test, placed in a real test directory, give `pass` and
   `fail` with the official command. If not: try the existing pytest path; if that fails too, mark
   `environment_invalid` and stop.
3. Record framework (`pytest`, `unittest`/`runtests.py`, sympy `bin/test`), test directories, the
   conftest files on the path, and the Python version.

## 3. The phases

Each phase lists: goal, internal stages (the planned order of work inside the phase), allowed tools,
budget, finish tool, gate, and where a failure goes. Budgets sum to 30.

### 3.1 Phase 0: Preflight (no model call)

- **Stages:** environment checks (§2) → extract code blocks and tracebacks from the issue →
  run each code block that looks runnable (with the official env, short timeout) and keep the
  output → build location candidates (§4).
- **Output:** `env`, `issue_snippets` (code, ran: bool, output, exception), `candidates`.
- **Cost:** no tokens. Often gives the reproduction for free.

### 3.2 Phase 1: Understand (budget 2, no repository tools)

- **Goal:** turn the issue into claims: what is observed, what is expected, and what is unknown.
- **Stages:** read issue → list observed behaviour with quotes → list expected behaviour with
  quotes → list what the issue does not say (unknowns) → note whether preflight already reproduced it.
- **Tools:** `finish_understand(observed=[{value, quote}], expected=[{value, quote}], unknowns=[{question, why}])`.
- **Gate:** every quote is found verbatim (whitespace-normalised) in the issue text; at least one
  observed claim. Expected claims may be empty; then each expected value becomes an unknown.
- **On fail:** repeat once with the gate's message ("this quote is not in the issue: ..."), then
  keep only the verified claims and continue.

### 3.3 Phase 2: Reproduce (budget 8)

- **Goal:** observe the bug on the buggy code.
- **Stages:** (a) if preflight ran an issue snippet and its output matches the observed claim, check
  and finish in one turn; (b) otherwise adapt the snippet (imports, setup) and run it; (c) otherwise
  write a minimal script from the description; (d) record the exact output.
- **Tools:** `read_file`, `grep`, `glob`, `run_python`, `finish_reproduce(script, output_summary)`.
- **Gate (symptom matcher, deterministic):** the controller re-runs `script` itself and compares the
  output with the observed claims: same exception class, or the observed value/message token appears
  in the output. Result: `matched`, `partial` (exception class differs, or only part of the message),
  `none`.
- **On fail:** stay in phase until the budget is spent; then continue in **unreproduced mode** with
  the flag `unreproduced` (§5).

### 3.4 Phase 3: Pin expected (budget 6)

- **Goal:** for each unknown, find evidence of the correct behaviour, or leave it flagged.
- **Stages:** for each unknown in order: (a) documentation and docstrings (`grep`/`read_file`);
  (b) existing tests of the same function; (c) a similar case that works today (`run_python`);
  (d) give up and keep it unknown.
- **Tools:** `read_file`, `grep`, `glob`, `run_python`, `finish_pin(resolved=[{unknown, value, source, quote, where}])`.
- **Gate:** for `source=docs|existing_test`, `quote` appears verbatim in the file at `where`; for
  `source=observed_run|analogy`, the controller re-runs the given snippet and finds `quote` in its
  output; `source=model` is accepted only as **unknown** (never verified).
- **Skip rule:** skipped when every expected claim from phase 1 is verified by an issue quote.
- **On fail:** unverified items stay unknowns. Never blocks.

### 3.5 Phase 4: Locate the test home (budget 3)

- **Goal:** choose where the test lives so the project's own runner and fixtures apply.
- **Stages:** check the ranked test-file candidates from preflight (§4) → read one to learn its
  style (pytest functions vs `TestCase`, fixtures used) → choose a directory and a module/label.
- **Tools:** `glob`, `grep`, `read_file`, `finish_locate(test_dir, style_notes)`. For Django:
  `module` as today (must exist in `HEAD`).
- **Gate:** the directory exists in `HEAD` (`git cat-file -e`), and the preflight trivial test runs
  there with the official command.
- **On fail:** fall back to the repository's default test directory (today's behaviour) with the flag
  `default_location`.

### 3.6 Phase 5: Write and run (budget 8)

- **Goal:** a test that fails on the buggy code at an assertion that checks the expected value.
- **Stages:** write the test using only verified expected values (unknown ones as a commented
  `# UNVERIFIED:` assertion or omitted, see decision D2) → run it → read the failure → fix the test
  (not the expectation) → finish.
- **Tools:** `read_file`, `run_test` (writes to the test home, official command), `finish_write(code, assertion_map=[{line, evidence_id}])`.
- **Gate (all deterministic):**
  1. outcome on buggy is `fail`, and failure reason is `assertion` (Django: `assertion` or `exception`, as today);
  2. the failing assertion's line is in `assertion_map` and points to verified evidence;
  3. the failure message contains the observed value from phase 2 when one exists (symptom check).
- **On fail, by cause:** passed on buggy → back to **Reproduce** (the bug was not triggered);
  crashed/import error → stay in **Write**; fails at an assertion without evidence → back to
  **Pin expected**; failure message does not match the symptom → back to **Reproduce**.

### 3.7 Phase 6: Finalise (no model call)

- Writes the submission and the evidence summary (as comments at the top of the test, and as fields
  in the run record), the flags, and the phase history. The submitted test is what SWT-bench and our
  L0–L4 labelling score.

### Budgets

| Phase | Turns | Skipped when |
|---|---|---|
| Understand | 2 | never |
| Reproduce | 8 | preflight already matched the symptom (then 1 turn) |
| Pin expected | 6 | all expected values verified from the issue |
| Locate | 3 | Django module given and valid, or one obvious candidate |
| Write | 8 | never |
| Reserve for sent-back phases | 3 | — |
| **Total** | **30** | |

Turns saved by skipped phases go to the reserve. A phase can be visited at most twice
(`MAX_VISITS`); after that the run degrades (§5).

## 4. Location: how we find code and test files

**Today.** The model searches freely (`grep` over `*.py`, `glob` over tracked files, `read_file`).
The test is placed for it (repo root, or a Django module it names). Nothing ranks candidates.

**Procedure agent: candidates computed in preflight (no model call), then confirmed by the model.**

| Signal from the issue | How it becomes a candidate |
|---|---|
| Traceback frames (`File "/.../django/db/models/query.py", line 673`) | Map to repository paths; the deepest frames inside the project are code candidates |
| Identifiers in code blocks and backticks (`Engine.render_to_string`, `KBinsDiscretizer`) | Look up definitions with a Python `ast` index of the repository (class/def → file, line) |
| Module paths and links to source lines | Direct |
| Error messages quoted in the issue | `grep` for the message string in the source |

**Test-file candidates** for each code candidate `pkg/sub/module.py`, in order:
1. existing tests that import or call the identifier (`grep` in test directories);
2. the conventional neighbour: `pkg/sub/tests/test_module.py`, `tests/<app>/` for Django,
   `sympy/<sub>/tests/test_<module>.py` for sympy;
3. the repository's default test directory.

The model sees the ranked lists in the handoff record; it confirms or changes them in the
*locate* phase. Both lists, and which entry the model picked, are stored, so we can measure later
how often the automatic ranking was right (useful on its own and cheap).

**Matching an issue to past examples (optional, Phase 1 variant):** the repository's own history
before `base_commit` contains earlier fixes with their tests. Retrieving the most similar earlier
test (by file and identifier overlap) as an example is gold-free and was the biggest single gain in
Qureshi & Jiang (open-book examples, up to 3× for weak models). It is listed as an optional variant,
not part of the first version.

## 5. When a run or a phase does not work

| Situation | Detected by | What happens | Recorded as |
|---|---|---|---|
| Container or Docker error | exception in `session.exec` | restart the container, repeat the tool call once; not counted against the model's budget | `infra_retry` |
| Environment cannot run tests | preflight | stop before any model call; excluded like today | `environment_invalid` |
| Issue code cannot run (needs network, GUI, other OS, missing optional package) | preflight / reproduce | continue in **unreproduced mode**: the test is written from verified expected claims; symptom checks are skipped | `unreproduced` |
| Bug not visible in this version | reproduce budget spent with symptom `none` | same as above | `unreproduced` |
| Expected value cannot be pinned | pin_expected | unknowns stay unknown; the assertion is commented or omitted (D2) | `unknown_expected` |
| Test never fails cleanly | write budget spent | submit the best attempt as today (forced submit) | `forced_submit` |
| A phase loops | `MAX_VISITS` | degrade: skip to *write* with what is known | `degraded:<phase>` |
| Tool output too long | `_clip` (2,500 chars) | keep the **end** of test output as well as the start (today the assertion line at the end can be cut) | — |

The flags are part of the result: flagged tests are expected to be wrong more often, and we measure
whether they are (the precision of the flags).

## 6. Prompts

Every phase receives the same short system prompt plus a phase prompt. The phase prompt includes
the handoff record rendered as text (not the previous transcripts).

**System (all phases):**
```
You are one step in a pipeline that writes a bug reproduction test for a GitHub issue in {repo}
at the buggy commit {base_commit}. The bug is NOT fixed in this checkout. You cannot see or run the
fixed code. You work only on the current step; a controller checks your result and moves on.
When the step is done, call {finish_tool} exactly once with the requested fields. Quotes must be
copied verbatim from their source. Never state a value as fact without its source; if you cannot
find a source, say so and it will be recorded as unknown. That is an acceptable outcome.
```

**Understand:**
```
STEP: understand the issue.
Read the issue below. Report:
1. observed: what goes wrong (wrong output, exception, crash). For each, quote the issue sentence or
   code/output that shows it.
2. expected: what the correct behaviour is, ONLY if the issue states or shows it (e.g. "should
   return", "I expected", "Expected behavior:", correct output shown). Quote it. Saying something
   is wrong is not an expected behaviour.
3. unknowns: what a test would need to assert but the issue does not state.
Do not look at the code in this step.
{issue}
{preflight_summary}
```

**Reproduce:**
```
STEP: reproduce the bug on the buggy code.
Observed behaviour to reproduce: {observed_claims}
{preflight: issue snippets and what they printed when run}
Write or adapt a short Python script that shows this behaviour, run it with run_python, and
finish with the script and what it printed. If the issue's example already shows it, use it as is.
If you cannot reproduce it within your steps, finish with what you tried and what happened.
Code candidates (from the issue's tracebacks and names): {code_candidates}
```

**Pin expected:**
```
STEP: establish the correct expected value for each unknown.
Unknowns: {unknowns}
For each, look for evidence in this order: documentation or docstrings; existing tests of the
same function; a similar case that works correctly today (run it). Report each value with its
source type, the verbatim quote, and the file or the snippet. If you find no evidence, leave it
unresolved. A guess without evidence is recorded as unknown, not as a value.
```

**Locate:**
```
STEP: choose where the test should live.
Test-file candidates (ranked): {test_candidates}
Pick the directory whose existing tests cover the code involved, read one test there to learn
the style (pytest functions or TestCase classes, fixtures), and finish with the directory
{django: and the test package name}.
```

**Write:**
```
STEP: write the test.
Reproduction: {repro_script} printed {repro_output}
Verified expected values: {verified_expected}   (each has an id)
Unknown values: {unknowns}   -> do not assert these as facts; {D2 rule}
Test home: {test_dir} ({style_notes})
Write one test that triggers the behaviour from the reproduction and asserts the verified
expected value(s). Run it with run_test. It must fail on this buggy code AT the assertion. If it
errors before the assertion, fix the test, not the expected value. Finish with the code and, for
each assertion line, the id of the evidence it relies on.
```

**Feedback when sent back** (examples, generated from the gate result):
- to Reproduce: "The test passed on the buggy code, so it did not trigger the bug. The reproduction
  printed {repro_output}; make the test do the same thing."
- to Pin expected: "The test fails at line {n}, but that assertion has no verified evidence."
- to Write: "The test crashed before its assertion: {error}. Fix the test setup."

## 7. Cost and how it is kept down

Measured today (stored runs): 87–97% of tokens are input (the transcript is re-sent every turn);
retries use 28% (nemotron) to 51% (gpt-4o-mini) of tokens because they restart from scratch;
median 17–20 turns per attempt.

Savings built into the design: fresh short context per phase (handoff record instead of
transcript); retries resume at the failed phase; preflight runs the issue's code without the model;
phases skip when already satisfied; deterministic gates cost nothing; the stable prompt prefix can
be cached by providers that support it. Estimate: about $0.02–0.04 per bug with nemotron, against
$0.026 today. To be measured in Phase 1; the comparison metric is **cost per valid test**.

## 8. What is recorded per run

Each run record (`results.jsonl`, `mode = "agent_procedure"`) adds: `phases` (per phase:
iterations, tokens, cost, gate result, visits), `record` (the final handoff record: claims,
evidence with `verified`, unknowns, repro, test home), `flags`, `candidates` (code and test, with
the chosen entry), and the `code` provenance already added on 2026-11-11.

## 9. Phase 1 experiment (development bugs, not the 114)

| Variant | What | Question |
|---|---|---|
| V0 | today's agent | baseline |
| V1 soft | same steps and prompts as one prompt, all tools, no gates | do the steps help? |
| V2 strict | the procedure agent | does enforcing them help more? |
| V2 − handoff | strict, but each phase sees the full transcript | how much of the cost saving is the handoff? |
| V2 + examples (optional) | strict plus retrieved earlier tests (§4) | do examples help? |

About 30 SWE-bench Lite bugs outside the 114, fixed seed, one run each first; measure L4, L3 share,
cost per valid test, turns, flags and flag precision. The best variant is frozen with a git tag and
becomes the "improved agent" of Phase 2.

## 10. Decisions still open

| # | Decision | Proposal |
|---|---|---|
| D1 | Strict, soft or both in Phase 1 | Both (cheap on 30 bugs) |
| D2 | Unverified expected values in the test | Omit the assertion and flag the test; keep the value as a comment |
| D3 | Total budget | 30 turns, as today |
| D4 | Official SWE-bench test command for every repository | Yes; keep today's pytest path as fallback |
| D5 | Symptom matcher: deterministic only, or with a cheap model check for `partial` | Deterministic only in the first version |
| D6 | Retrieved examples from repository history | Later variant, not in the first version |
| D7 | Model | nemotron-3-super (paid) for Phase 1; gpt-4o-mini optional |

## 11. Implementation map (current code → change)

| File | Change |
|---|---|
| `src/tdbf/agent/procedure.py` | new: controller, `Record`, phase specs, gates |
| `src/tdbf/agent/tools.py` | add `finish_*` tools; let tools be gated per phase; `run_test` takes a test home |
| `src/tdbf/harness/runner.py` | adapter using `MAP_REPO_VERSION_TO_SPECS[...]["test_cmd"]`; place tests in the chosen directory |
| `src/tdbf/agent/preflight.py` | new: env checks, issue snippet extraction and runs, candidate ranking (`ast` index) |
| `src/tdbf/agent/pipeline.py` | `run_procedure` next to `run_brt_phase`; same record format plus the new fields |
| `scripts/16_nightly_batch.py` | `--agent procedure`; reuse each image across repeats and conditions |
| `src/tdbf/studies.py` | register the Phase 1 development run groups before any run |
| `scripts/*_smoke.py` | offline tests for gates (quote check, symptom matcher, assertion map) without Docker |
