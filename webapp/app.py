"""TDBF Flask webapp — BRT results dashboard."""

from __future__ import annotations

import json
from pathlib import Path

from flask import (
    Flask,
    abort,
    jsonify,
    redirect,
    render_template,
    request,
    url_for,
)

import db
import glossary
import research_data

from tdbf import ledger as tdbf_ledger
from tdbf import metrics as tdbf_metrics

app = Flask(__name__)
app.jinja_env.globals["GLOSSARY"] = glossary.TERMS
db.init_db()

RESULTS_FILE = Path(__file__).parent.parent / "results.jsonl"
SWT_FILE = RESULTS_FILE.parent / "swt_results.jsonl"
PUBLISHED_FILE = RESULTS_FILE.parent / "published_swt_lite.jsonl"

_LEVEL_NAMES = {
    0: "L0 Malformed",
    1: "L1 Non-reproducing",
    2: "L2 Crashes",
    3: "L3 Candidate",
    4: "L4 Validated",
}

_LEVEL_COLORS = {
    0: "secondary",
    1: "danger",
    2: "warning",
    3: "info",
    4: "success",
}


def _c2p(l3: int, l4: int) -> str:
    if l3 + l4 == 0:
        return "—"
    return f"{l4 / (l3 + l4):.0%}"


# ── pages ─────────────────────────────────────────────────────────────────────

def _rate(num: int, denom: int) -> str:
    return f"{num / denom:.0%}" if denom else "—"


@app.route("/")
def index():
    stats = db.get_dashboard_stats()
    fix = db.get_fix_stats()
    level_map = {r["best_level"]: r["cnt"] for r in stats["by_level"]}

    for row in stats["by_model"]:
        row["c2p"] = _c2p(
            row["candidates"] - row["l4"],
            row["l4"],
        )
    for row in fix["by_model"]:
        row["rate"] = _rate(row["resolved"], row["total"])
    for row in fix["by_repo"]:
        row["rate"] = _rate(row["resolved"], row["total"])

    overall_l3 = level_map.get(3, 0)
    overall_l4 = level_map.get(4, 0)
    return render_template(
        "index.html",
        total=stats["total"],
        brt_total=stats["brt_total"],
        level_map=level_map,
        level_names=_LEVEL_NAMES,
        level_colors=_LEVEL_COLORS,
        by_model=stats["by_model"],
        overall_c2p=_c2p(overall_l3, overall_l4),
        fix=fix,
        fix_rate=_rate(fix["resolved"], fix["total"]),
    )


@app.route("/runs")
def runs():
    model_filter = request.args.get("model") or None
    level_filter = request.args.get("level")
    mode_filter = request.args.get("mode") or None
    level_int = int(level_filter) if level_filter and level_filter.isdigit() else None

    rows = db.get_runs(model=model_filter, level=level_int, mode=mode_filter)
    models = db.get_all_models()
    return render_template(
        "runs.html",
        runs=rows,
        models=models,
        level_names=_LEVEL_NAMES,
        level_colors=_LEVEL_COLORS,
        selected_model=model_filter or "",
        selected_level=level_filter or "",
        selected_mode=mode_filter or "",
    )


@app.route("/commands")
def commands():
    return render_template("commands.html")


def _frozen_sample() -> tuple[list[str] | None, dict[str, str]]:
    """The frozen sample's instance ids and each one's environment state, so the
    Results page can say which instances were left out and why."""
    # Every registered sample (tdbf.ledger.SAMPLE_FILES), so an extension sample
    # shows up here as soon as it is added there.
    samples, env = tdbf_ledger.load_samples(RESULTS_FILE.parent)
    if not samples:
        return None, {}
    ids = [i for s in samples for i in s.get("instances") or []]
    states = {k: (v or {}).get("state", "?") for k, v in env.items()}
    return ids, states


@app.route("/results")
def results():
    """The main experiment only: one model, the frozen sample, both arms.
    Numbers come from tdbf.metrics.build_results (also behind script 14)."""
    recs = tdbf_metrics.load_records(RESULTS_FILE)
    ids, states = _frozen_sample()
    # ?view=original: Django test-first from the first runs (no rq1-djv2
    # substitution). The other view's resolved counts go in the note.
    original = request.args.get("view") == "original"
    views = {False: tdbf_metrics.EXPERIMENT_ARMS, True: tdbf_metrics.EXPERIMENT_ARMS_ORIGINAL}
    res = tdbf_metrics.build_results(recs, sample_ids=ids, env_states=states,
                                     arms_def=views[original])
    other = tdbf_metrics.build_results(recs, sample_ids=ids, env_states=states,
                                       arms_def=views[not original])
    return render_template("results.html", res=res, other=other, original=original,
                           glossary_sections=glossary.RESULTS_SECTIONS)


@app.route("/swt")
def swt():
    """Our stored test-first tests re-scored by the official SWT-bench harness,
    next to other tools' published per-instance results. Numbers come from
    tdbf.metrics.build_swt; the records from scripts/17_swt_predictions.py ingest."""
    res = tdbf_metrics.build_swt(tdbf_metrics.load_records(SWT_FILE),
                                 tdbf_metrics.load_records(PUBLISHED_FILE))
    return render_template("swt.html", res=res,
                           glossary_sections=glossary.sections_for(glossary.SWT_TERMS))


@app.route("/development")
def development():
    """Everything run while building the agents. The experiment's own run groups
    are left out, so no number here repeats one from the Results page."""
    recs = [r for r in tdbf_metrics.load_records(RESULTS_FILE)
            if (r.get("flags") or {}).get("run_group")
            not in tdbf_metrics.EXPERIMENT_RUN_GROUPS]
    return render_template("development.html", rep=tdbf_metrics.build_report(recs),
                           glossary_sections=glossary.DEVELOPMENT_SECTIONS)


@app.route("/status")
def status():
    """Where every SWE-bench Lite instance stands: run, pending, excluded, or
    never drawn. Same table as scripts/15_status.py (tdbf.ledger)."""
    led = tdbf_ledger.load_ledger(RESULTS_FILE.parent)
    status_filter = request.args.get("status") or ""
    repo_filter = request.args.get("repo") or ""
    rows = [r for r in led["rows"]
            if (not status_filter or r["status"] == status_filter)
            and (not repo_filter or r["repo"] == repo_filter)]
    return render_template(
        "status.html", led=led, rows=rows, statuses=tdbf_ledger.STATUSES,
        status_filter=status_filter, repo_filter=repo_filter,
        glossary_sections=glossary.sections_for(glossary.STATUS_TERMS))


@app.route("/metrics")
def metrics():
    """Old address of the metrics page; the experiment now lives on /results."""
    return redirect(url_for("results"))


@app.route("/research")
def research():
    """Research hub: plan status, decisions log, related-work comparison.
    Content lives in research_data.py (git-versioned, reviewable in diffs)."""
    return render_template(
        "research.html",
        questions=research_data.RESEARCH_QUESTIONS,
        scope=research_data.SCOPE_CONDITIONS,
        superseded=research_data.SUPERSEDED_RQ_MAP,
        plan=research_data.RESEARCH_PLAN,
        decisions=research_data.DECISIONS,
        related=research_data.RELATED_WORK,
    )


# ── docs / research log ─────────────────────────────────────────────────────

def _group_docs_by_topic(docs: list[dict]) -> list[tuple[str, list[dict]]]:
    grouped: dict[str, list[dict]] = {}
    for d in docs:
        grouped.setdefault(d["topic"] or "Uncategorised", []).append(d)
    return list(grouped.items())


@app.route("/docs")
def docs_index():
    docs = db.list_docs()
    return render_template(
        "docs.html",
        groups=_group_docs_by_topic(docs),
        count=len(docs),
    )


@app.route("/docs/new", methods=["GET", "POST"])
def doc_new():
    if request.method == "POST":
        title = (request.form.get("title") or "").strip()
        if not title:
            abort(400, "title is required")
        doc_id = db.create_doc(
            title=title,
            body=request.form.get("body") or "",
            topic=(request.form.get("topic") or "").strip(),
            pinned=bool(request.form.get("pinned")),
        )
        return redirect(url_for("doc_detail", slug=db.get_doc(doc_id)["slug"]))
    return render_template(
        "doc_form.html", doc=None, topics=db.list_topics(), action=url_for("doc_new")
    )


@app.route("/docs/<slug>")
def doc_detail(slug: str):
    doc = db.get_doc_by_slug(slug)
    if not doc:
        abort(404)
    return render_template("doc_detail.html", doc=doc)


@app.route("/docs/<int:doc_id>/edit", methods=["GET", "POST"])
def doc_edit(doc_id: int):
    doc = db.get_doc(doc_id)
    if not doc:
        abort(404)
    if request.method == "POST":
        title = (request.form.get("title") or "").strip()
        if not title:
            abort(400, "title is required")
        db.update_doc(
            doc_id,
            title=title,
            body=request.form.get("body") or "",
            topic=(request.form.get("topic") or "").strip(),
            pinned=bool(request.form.get("pinned")),
        )
        return redirect(url_for("doc_detail", slug=db.get_doc(doc_id)["slug"]))
    return render_template(
        "doc_form.html", doc=doc, topics=db.list_topics(),
        action=url_for("doc_edit", doc_id=doc_id),
    )


@app.route("/docs/<int:doc_id>/delete", methods=["POST"])
def doc_delete(doc_id: int):
    db.delete_doc(doc_id)
    return redirect(url_for("docs_index"))


@app.route("/runs/<int:run_id>")
def run_detail(run_id: int):
    run = db.get_run(run_id)
    if not run:
        abort(404)
    return render_template(
        "run_detail.html",
        run=run,
        level_names=_LEVEL_NAMES,
        level_colors=_LEVEL_COLORS,
    )


# ── API ───────────────────────────────────────────────────────────────────────

@app.route("/api/runs", methods=["POST"])
def api_save_run():
    record = request.get_json(force=True)
    if not record or "instance_id" not in record:
        return jsonify({"error": "invalid payload"}), 400
    run_id = db.insert_run(record)
    return jsonify({"id": run_id}), 201


# ── CLI: flask sync ───────────────────────────────────────────────────────────

@app.cli.command("sync")
def sync_from_jsonl():
    """Backfill SQLite from results.jsonl (skips already-imported records)."""
    if not RESULTS_FILE.exists():
        print("results.jsonl not found — nothing to sync.")
        return
    added = skipped = 0
    with RESULTS_FILE.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if db.already_exists(
                record.get("instance_id", ""),
                record.get("model", ""),
                record.get("timestamp", ""),
            ):
                skipped += 1
            else:
                db.insert_run(record)
                added += 1
    print(f"Sync complete: {added} added, {skipped} skipped.")


if __name__ == "__main__":
    app.run(debug=True)
