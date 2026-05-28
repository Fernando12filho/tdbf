"""TDBF Flask webapp — BRT results dashboard."""

from __future__ import annotations

import json
from pathlib import Path

from flask import Flask, abort, jsonify, render_template, request

import db

app = Flask(__name__)
db.init_db()

RESULTS_FILE = Path(__file__).parent.parent / "results.jsonl"

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

@app.route("/")
def index():
    stats = db.get_dashboard_stats()
    level_map = {r["best_level"]: r["cnt"] for r in stats["by_level"]}

    for row in stats["by_model"]:
        row["c2p"] = _c2p(
            row["candidates"] - row["l4"],
            row["l4"],
        )

    overall_l3 = level_map.get(3, 0)
    overall_l4 = level_map.get(4, 0)
    return render_template(
        "index.html",
        total=stats["total"],
        level_map=level_map,
        level_names=_LEVEL_NAMES,
        level_colors=_LEVEL_COLORS,
        by_model=stats["by_model"],
        overall_c2p=_c2p(overall_l3, overall_l4),
    )


@app.route("/runs")
def runs():
    model_filter = request.args.get("model") or None
    level_filter = request.args.get("level")
    level_int = int(level_filter) if level_filter and level_filter.isdigit() else None

    rows = db.get_runs(model=model_filter, level=level_int)
    models = db.get_all_models()
    return render_template(
        "runs.html",
        runs=rows,
        models=models,
        level_names=_LEVEL_NAMES,
        level_colors=_LEVEL_COLORS,
        selected_model=model_filter or "",
        selected_level=level_filter or "",
    )


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
