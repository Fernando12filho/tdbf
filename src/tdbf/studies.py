"""Study registry: which run groups form which study, made with which model.

`results.jsonl` holds every run since May 2026 — Claude and Ollama development
runs, pilots, the RQ1 experiment, and tests written by other models — told apart
only by `flags.run_group`. Before this registry the experiment was selected by
run-group name alone and the model was never checked, so a run written into an
experiment group by another model (or a later re-run) would have silently taken
the place of the experiment's own records.

A study names its run groups and the exact model spec(s) allowed in them. Two
rules follow, enforced in two places:

  - read side (`records_of`): a study's numbers are computed only from its own
    records, and a record in its run groups made by another model is an error,
    never a silent mix;
  - write side (`check_write`, called by `results.save_run` and, before any model
    call, by `results.use_run_group`): a new record must name a registered run
    group, the group's study must be open, and the model must be the study's.

Adding a run: register its run group here first (a new study, or an open one).
Closing a study: set `frozen=True`; its records can then never change.

The log stays append-only; nothing here rewrites it. Records from before run
groups existed (`run_group` None) belong to the frozen `dev-legacy` study.
"""

from __future__ import annotations

from dataclasses import dataclass

NEMOTRON_FREE = "openrouter:nvidia/nemotron-3-super-120b-a12b:free"
GPT_4O_MINI = "openrouter:openai/gpt-4o-mini-2024-07-18@openai"
GPT_5_MINI = "openrouter:openai/gpt-5-mini@openai"
GPT_5_NANO = "openrouter:openai/gpt-5-nano@openai"

PURPOSES = ("experiment", "pilot", "development")


@dataclass(frozen=True)
class Study:
    name: str
    purpose: str                       # one of PURPOSES
    run_groups: tuple[str | None, ...]
    models: tuple[str, ...]            # exact client names (`client.name`)
    frozen: bool
    description: str
    samples: tuple[str, ...] = ()      # frozen sample files, if any

    def __post_init__(self) -> None:
        if self.purpose not in PURPOSES:
            raise ValueError(f"study {self.name}: purpose {self.purpose!r} not in {PURPOSES}")


STUDIES: tuple[Study, ...] = (
    Study(
        name="rq1-nemotron", purpose="experiment",
        run_groups=("rq1-sweep", "rq1-armA", "rq1-ext", "rq1-ext-v2", "rq1-djv2"),
        models=(NEMOTRON_FREE,), frozen=True,
        samples=("sample_lite_frozen.json", "sample_lite_ext_frozen.json"),
        description="RQ1: test-first vs. reactive on the frozen samples, nemotron "
                    "(free tier, served by Nvidia). Frozen 2026-11-03: the free id was "
                    "withdrawn 10-27, so any continuation is a new study."),
    Study(
        name="test-agent-gpt-4o-mini", purpose="experiment",
        run_groups=("tests-4omini",), models=(GPT_4O_MINI,), frozen=False,
        samples=("sample_tests_swt114_frozen.json",),
        description="Test agent only (Phase A) on GPT-4o-mini 2024-07-18, Issue2Test's "
                    "model, on the instances of rq1-nemotron that are in SWT-bench Lite. "
                    "Scored by the SWT-bench harness."),
    Study(
        name="test-agent-gpt-4o-mini-pilot", purpose="pilot",
        run_groups=("swt-4omini-pilot",), models=(GPT_4O_MINI,), frozen=True,
        description="5-instance pilot of test-agent-gpt-4o-mini (2026-11-03)."),
    Study(
        name="nemotron-pilots", purpose="pilot",
        run_groups=("v2-pilot", "probe-0921", "validate-sweep"),
        models=(NEMOTRON_FREE,), frozen=True,
        description="Capability probe, sweep validation and the Django-native BRT pilot "
                    "on nemotron, before and around RQ1."),
    Study(
        name="scouts-dev", purpose="development",
        run_groups=("scouts-dev-match", "scouts-dev-full", "scouts-dev-full-s100"),
        models=(GPT_5_MINI,), frozen=False,
        samples=("studies/scouts/data/dev_bugs.json",),
        description="Scouts architecture for Phase A (agent/scouts): three scouts in "
                    "parallel (Test Template Generator, Edit Localizer, Issue Reproducer), "
                    "an Organizer, then the BRT agent with gate 2 (the failure must match "
                    "the issue, judged gold-free, retried with feedback). Two arms on the 30 "
                    "development bugs of the procedure study: scouts-dev-match = the "
                    "unchanged BRT agent + gate 2 only; scouts-dev-full = scouts + "
                    "organizer + gate 2. Same model for every role (gpt-5-mini), 50 turns "
                    "per writer attempt, as the procedure study's baseline (21/30 L4), "
                    "which is the reference. scouts-dev-full stopped after 7 bugs with scouts "
                    "capped at 20 turns (most hit the cap); scouts-dev-full-s100 re-runs "
                    "the full arm on all 30 with scouts capped at 100 turns (user decision "
                    "2026-11-24). Development results, reported as such."),
    Study(
        name="scouts-dev-v2", purpose="development",
        run_groups=("v2-base", "v2-loop", "v2-full", "v2-full-1m"),
        models=(GPT_5_MINI, GPT_5_NANO), frozen=False,
        samples=("studies/scouts/data/dev_round4.json", "studies/scouts/data/dev_round5.json"),
        description="Scouts v2 (studies/scouts/plan_v2.md) on 20 fresh development bugs "
                    "(rounds 4-5 of the procedure study's order): v2-base = the old BRT agent; "
                    "v2-loop = the BRT agent with judge v2 inside its loop (check_test + "
                    "guarded submit_test); v2-full = gpt-5-nano scouts and organizer, then the "
                    "same writer loop. Writer and judge gpt-5-mini. v2-full holds only the "
                    "1-bug live check with scouts capped at 300K tokens; the arm runs as "
                    "v2-full-1m (1M tokens, amendment 1 of plan_v2.md). Development results."),
    Study(
        name="blackboard-dev", purpose="development",
        run_groups=("bb-fresh", "bb-dev30", "rbb-fresh", "rbb-dev30"),
        models=(GPT_5_MINI, GPT_5_NANO), frozen=False,
        samples=("studies/scouts/data/dev_round4.json", "studies/scouts/data/dev_round5.json",
                 "studies/scouts/data/dev_bugs.json"),
        description="Blackboard Phase A (studies/scouts/plan_blackboard.md): gpt-5-nano scouts "
                    "and the gpt-5-mini in-loop-judge writer work concurrently and share "
                    "findings on a structured board; no organizer. bb-fresh = the 20 fresh "
                    "bugs (compared with v2-base/loop/full-1m), bb-dev30 = the 30 development "
                    "bugs (compared with the stored v1 results). rbb-* = the reactive "
                    "blackboard (plan_reactive_blackboard.md) on the same bugs. Development "
                    "results."),
    Study(
        name="final-swt-lite", purpose="experiment",
        run_groups=("final-single", "final-rbb"),
        models=(GPT_5_MINI, GPT_5_NANO), frozen=False,
        samples=("studies/final-eval/data/sample_swt_lite_final.json",),
        description="Final evaluation (studies/final-eval/plan.md): TDBF's single test-writing "
                    "agent (final-single) and the reactive blackboard (final-rbb) on the 226 "
                    "clean SWT-bench Lite bugs, scored with the official SWT-bench harness next "
                    "to ReProAgent's re-scored GPT-5-mini predictions. The 50 development bugs "
                    "reuse proc-dev-baseline/v2-base and rbb-dev30/rbb-fresh."),
    Study(
        name="multitest-dev", purpose="development",
        run_groups=("mt-brt",),
        models=(GPT_5_MINI,), frozen=False,
        samples=("studies/scouts/data/dev_bugs.json", "studies/scouts/data/dev_round4.json",
                 "studies/scouts/data/dev_round5.json"),
        description="Multi-test BRTs (studies/multi-test/plan.md) on the 50 development bugs, "
                    "gpt-5-mini: mt-brt = the single agent told to write a main test plus as "
                    "many related-case tests as it judges the bug needs; compared with the "
                    "stored one-test tests (SWT-bench S, coverage, L4). No fix runs. "
                    "Development results."),
    Study(
        name="dev-sonnet-gate-fix", purpose="development",
        run_groups=("gate-fix", "gate-fix-aborted"),
        models=("anthropic:claude-sonnet-4-6",), frozen=True,
        description="Sonnet re-run of the 14-instance development set after the gate "
                    "fix (2026-09-09)."),
    Study(
        name="dev-legacy", purpose="development",
        run_groups=(None,),
        models=("anthropic:claude-sonnet-4-6", "anthropic:claude-opus-4-7",
                "anthropic:claude-fable-5", "ollama:qwen2.5-coder:7b-instruct-q4_K_M",
                "ollama:qwen2.5-coder:14b-instruct-q3_K_M", "ollama:qwen3.5:9b",
                "ollama:mistral:7b"),
        frozen=True,
        description="Development runs from before run groups existed (May-Sep 2026): "
                    "Claude and local Ollama models, agent phases built one at a time."),
)

_BY_NAME = {s.name: s for s in STUDIES}
_BY_GROUP: dict[str | None, Study] = {}
for _s in STUDIES:
    for _g in _s.run_groups:
        if _g in _BY_GROUP:
            raise ValueError(f"run group {_g!r} is in two studies: "
                             f"{_BY_GROUP[_g].name} and {_s.name}")
        _BY_GROUP[_g] = _s


class StudyError(ValueError):
    """A record that breaks the registry: wrong model, unknown or frozen group."""


def get(name: str) -> Study:
    try:
        return _BY_NAME[name]
    except KeyError:
        raise StudyError(f"no study named {name!r}; known: {sorted(_BY_NAME)}") from None


def of_name(name: str | None) -> Study | None:
    return _BY_NAME.get(name)


def of_group(run_group: str | None) -> Study | None:
    return _BY_GROUP.get(run_group)


def run_group_of(rec: dict) -> str | None:
    return (rec.get("flags") or {}).get("run_group")


def of_record(rec: dict) -> Study | None:
    return of_group(run_group_of(rec))


def violations(records: list[dict]) -> list[str]:
    """Every record that does not fit the registry, as readable lines."""
    out = []
    for r in records:
        g = run_group_of(r)
        s = of_group(g)
        where = f"{r.get('instance_id')} ({r.get('mode')}, {r.get('timestamp', '')[:19]})"
        if s is None:
            out.append(f"{where}: run group {g!r} is not registered")
        elif r.get("model") not in s.models:
            out.append(f"{where}: model {r.get('model')!r} is not {s.name}'s "
                       f"({', '.join(s.models)})")
    return out


def records_of(name: str, records: list[dict]) -> list[dict]:
    """The study's records. A record in its run groups by another model raises:
    a study's numbers never mix models silently."""
    s = get(name)
    mine = [r for r in records if run_group_of(r) in s.run_groups]
    bad = [r for r in mine if r.get("model") not in s.models]
    if bad:
        lines = violations(bad)
        raise StudyError(f"study {name}: {len(bad)} record(s) by another model:\n  "
                         + "\n  ".join(lines[:10]))
    return mine


def check_write(run_group: str | None, model: str) -> Study:
    """May a new record be written to `run_group` by `model`? Returns the study,
    raises StudyError with what to do otherwise."""
    if not run_group:
        raise StudyError(
            "every new record needs a run group (--run-group); register it in "
            "src/tdbf/studies.py first")
    s = of_group(run_group)
    if s is None:
        raise StudyError(
            f"run group {run_group!r} is not registered: add it to a study in "
            "src/tdbf/studies.py (a new one, or an open one with this model)")
    if s.frozen:
        raise StudyError(
            f"run group {run_group!r} belongs to {s.name}, which is frozen: its "
            "records cannot change. Use a run group of an open study.")
    if model not in s.models:
        raise StudyError(
            f"run group {run_group!r} belongs to {s.name}, made with "
            f"{', '.join(s.models)}, not {model!r}. One study, one model.")
    return s
