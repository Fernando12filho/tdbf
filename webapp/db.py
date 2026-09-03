"""SQLite database layer for the TDBF webapp."""

from __future__ import annotations

import json
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path(__file__).parent / "tdbf.db"


def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


# Columns added after the original BRT-only schema. Kept as an idempotent
# migration so an existing tdbf.db picks them up without a rebuild.
#   runs.mode       'brt' (Phase A / legacy) | 'fix_agent' (Phase B)
#   runs.resolved   fix runs: 1 if the held-out tests passed, else 0
#   runs.cost_usd   total USD across attempts
_RUNS_MIGRATIONS = {
    "mode": "TEXT DEFAULT 'brt'",
    "resolved": "INTEGER",
    "cost_usd": "REAL",
}
_ATTEMPTS_MIGRATIONS = {
    "mode": "TEXT",
    "resolved": "INTEGER",
    "applied": "INTEGER",
    "stop_cause": "TEXT",
    "iterations": "INTEGER",
    "num_tool_calls": "INTEGER",
    "cost_usd": "REAL",
    "fix_patch": "TEXT",
    "fail_to_pass_failed": "TEXT",
    "pass_to_pass_failed": "TEXT",
}


def _migrate(conn, table: str, cols: dict[str, str]) -> None:
    existing = {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}
    for name, decl in cols.items():
        if name not in existing:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {decl}")


def init_db() -> None:
    with get_conn() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS runs (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                instance_id TEXT NOT NULL,
                repo        TEXT,
                model       TEXT NOT NULL,
                split       TEXT DEFAULT 'lite',
                think       INTEGER DEFAULT 0,
                context     INTEGER DEFAULT 0,
                observe     INTEGER DEFAULT 0,
                mode        TEXT DEFAULT 'brt',
                resolved    INTEGER,
                cost_usd    REAL,
                best_level  INTEGER,
                best_attempt INTEGER,
                total_attempts INTEGER,
                timestamp   TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS attempts (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id          INTEGER NOT NULL REFERENCES runs(id),
                attempt_number  INTEGER NOT NULL,
                level           INTEGER,
                mode            TEXT,
                resolved        INTEGER,
                applied         INTEGER,
                stop_cause      TEXT,
                iterations      INTEGER,
                num_tool_calls  INTEGER,
                cost_usd        REAL,
                fix_patch       TEXT,
                fail_to_pass_failed TEXT,
                pass_to_pass_failed TEXT,
                outcome_buggy   TEXT,
                outcome_fixed   TEXT,
                test_code       TEXT,
                probe_code      TEXT,
                observed_output TEXT,
                raw_response    TEXT,
                prompt_sent     TEXT,
                probe_prompt    TEXT,
                input_tokens    INTEGER,
                output_tokens   INTEGER,
                duration_seconds REAL
            );

            CREATE TABLE IF NOT EXISTS docs (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                slug        TEXT UNIQUE NOT NULL,
                title       TEXT NOT NULL,
                topic       TEXT,
                body        TEXT NOT NULL,
                pinned      INTEGER DEFAULT 0,
                created_at  TEXT NOT NULL,
                updated_at  TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_runs_instance ON runs(instance_id);
            CREATE INDEX IF NOT EXISTS idx_runs_model    ON runs(model);
            CREATE INDEX IF NOT EXISTS idx_runs_level    ON runs(best_level);
            CREATE INDEX IF NOT EXISTS idx_docs_topic    ON docs(topic);
        """)
        _migrate(conn, "runs", _RUNS_MIGRATIONS)
        _migrate(conn, "attempts", _ATTEMPTS_MIGRATIONS)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_runs_mode ON runs(mode)")
    seed_docs()


def _maybe_json(value):
    """Serialise list/dict values (e.g. failing-test lists) for TEXT columns."""
    if value is None or isinstance(value, str):
        return value
    return json.dumps(value)


def insert_run(record: dict) -> int:
    flags = record.get("flags", {})
    attempts_data = record.get("attempts", [])
    mode = record.get("mode") or "brt"
    resolved = record.get("resolved")
    total_cost = sum(
        (a.get("cost_usd") or 0) for a in attempts_data
    ) or None

    with get_conn() as conn:
        cur = conn.execute(
            """
            INSERT INTO runs
                (instance_id, repo, model, split, think, context, observe,
                 mode, resolved, cost_usd,
                 best_level, best_attempt, total_attempts, timestamp)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                record.get("instance_id", ""),
                record.get("repo", ""),
                record.get("model", ""),
                record.get("split", "lite"),
                int(flags.get("think", False)),
                int(flags.get("context", False)),
                int(flags.get("observe", False)),
                mode,
                int(resolved) if resolved is not None else None,
                total_cost,
                record.get("best_level"),
                record.get("best_attempt"),
                len(attempts_data),
                record.get("timestamp", ""),
            ),
        )
        run_id = cur.lastrowid

        for a in attempts_data:
            a_resolved = a.get("resolved")
            a_applied = a.get("applied")
            conn.execute(
                """
                INSERT INTO attempts
                    (run_id, attempt_number, level, mode, resolved, applied,
                     stop_cause, iterations, num_tool_calls, cost_usd, fix_patch,
                     fail_to_pass_failed, pass_to_pass_failed,
                     outcome_buggy, outcome_fixed,
                     test_code, probe_code, observed_output, raw_response,
                     prompt_sent, probe_prompt, input_tokens, output_tokens, duration_seconds)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    run_id,
                    a.get("attempt", 0),
                    a.get("level"),
                    a.get("mode"),
                    int(a_resolved) if a_resolved is not None else None,
                    int(a_applied) if a_applied is not None else None,
                    a.get("stop_cause"),
                    a.get("iterations"),
                    a.get("num_tool_calls"),
                    a.get("cost_usd"),
                    a.get("fix_patch"),
                    _maybe_json(a.get("fail_to_pass_failed")),
                    _maybe_json(a.get("pass_to_pass_failed")),
                    a.get("outcome_buggy"),
                    a.get("outcome_fixed"),
                    a.get("test_code"),
                    a.get("probe_code"),
                    a.get("observed_output"),
                    a.get("raw_response"),
                    a.get("prompt_sent"),
                    a.get("probe_prompt"),
                    a.get("input_tokens"),
                    a.get("output_tokens"),
                    a.get("duration_seconds"),
                ),
            )
    return run_id


def already_exists(instance_id: str, model: str, timestamp: str) -> bool:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT id FROM runs WHERE instance_id=? AND model=? AND timestamp=?",
            (instance_id, model, timestamp),
        ).fetchone()
    return row is not None


# ── queries ───────────────────────────────────────────────────────────────────

# BRT-class runs = Phase A + legacy workflow (everything that is not a fix run).
_BRT_WHERE = "mode IS NULL OR mode != 'fix_agent'"


def get_dashboard_stats() -> dict:
    with get_conn() as conn:
        total = conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
        by_level = conn.execute(
            f"SELECT best_level, COUNT(*) as cnt FROM runs WHERE {_BRT_WHERE} "
            "GROUP BY best_level ORDER BY best_level"
        ).fetchall()
        by_model = conn.execute(
            f"""
            SELECT model,
                   COUNT(*) as total,
                   SUM(CASE WHEN best_level=4 THEN 1 ELSE 0 END) as l4,
                   SUM(CASE WHEN best_level>=3 THEN 1 ELSE 0 END) as candidates
            FROM runs WHERE {_BRT_WHERE} GROUP BY model ORDER BY model
            """
        ).fetchall()
        brt_total = conn.execute(
            f"SELECT COUNT(*) FROM runs WHERE {_BRT_WHERE}"
        ).fetchone()[0]
    return {
        "total": total,
        "brt_total": brt_total,
        "by_level": [dict(r) for r in by_level],
        "by_model": [dict(r) for r in by_model],
    }


def get_fix_stats() -> dict:
    """Phase B (fix agent) resolved-rate, overall and per model / repo."""
    with get_conn() as conn:
        total = conn.execute(
            "SELECT COUNT(*) FROM runs WHERE mode='fix_agent'"
        ).fetchone()[0]
        resolved = conn.execute(
            "SELECT COUNT(*) FROM runs WHERE mode='fix_agent' AND resolved=1"
        ).fetchone()[0]
        cost = conn.execute(
            "SELECT COALESCE(SUM(cost_usd),0) FROM runs WHERE mode='fix_agent'"
        ).fetchone()[0]
        by_model = conn.execute(
            """
            SELECT model,
                   COUNT(*) as total,
                   SUM(CASE WHEN resolved=1 THEN 1 ELSE 0 END) as resolved,
                   COALESCE(SUM(cost_usd),0) as cost
            FROM runs WHERE mode='fix_agent' GROUP BY model ORDER BY model
            """
        ).fetchall()
        by_repo = conn.execute(
            """
            SELECT repo,
                   COUNT(*) as total,
                   SUM(CASE WHEN resolved=1 THEN 1 ELSE 0 END) as resolved
            FROM runs WHERE mode='fix_agent' GROUP BY repo ORDER BY repo
            """
        ).fetchall()
    return {
        "total": total,
        "resolved": resolved,
        "cost": cost,
        "by_model": [dict(r) for r in by_model],
        "by_repo": [dict(r) for r in by_repo],
    }


def get_runs(model: str | None = None, level: int | None = None,
             mode: str | None = None, limit: int = 100) -> list[dict]:
    clauses, params = [], []
    if model:
        clauses.append("model = ?")
        params.append(model)
    if level is not None:
        clauses.append("best_level = ?")
        params.append(level)
    if mode == "fix_agent":
        clauses.append("mode = 'fix_agent'")
    elif mode == "brt":
        clauses.append("(mode IS NULL OR mode != 'fix_agent')")
    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
    params.append(limit)

    with get_conn() as conn:
        rows = conn.execute(
            f"SELECT * FROM runs {where} ORDER BY timestamp DESC LIMIT ?",
            params,
        ).fetchall()
    return [dict(r) for r in rows]


def get_run(run_id: int) -> dict | None:
    with get_conn() as conn:
        run = conn.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
        if not run:
            return None
        attempts = conn.execute(
            "SELECT * FROM attempts WHERE run_id=? ORDER BY attempt_number",
            (run_id,),
        ).fetchall()
    return {**dict(run), "attempts": [dict(a) for a in attempts]}


def get_all_models() -> list[str]:
    with get_conn() as conn:
        rows = conn.execute("SELECT DISTINCT model FROM runs ORDER BY model").fetchall()
    return [r[0] for r in rows]


# ── docs (research log) ─────────────────────────────────────────────────────

def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")


def slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug or "doc"


def _unique_slug(conn, base: str, exclude_id: int | None = None) -> str:
    """Return ``base`` (or base-2, base-3, …) not used by another doc."""
    slug, n = base, 1
    while True:
        row = conn.execute(
            "SELECT id FROM docs WHERE slug=? AND id IS NOT ?",
            (slug, exclude_id),
        ).fetchone()
        if row is None:
            return slug
        n += 1
        slug = f"{base}-{n}"


def list_docs() -> list[dict]:
    """All docs, ready to group by topic in the template."""
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT id, slug, title, topic, pinned, created_at, updated_at "
            "FROM docs ORDER BY topic IS NULL, topic, pinned DESC, title"
        ).fetchall()
    return [dict(r) for r in rows]


def list_topics() -> list[str]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT DISTINCT topic FROM docs WHERE topic IS NOT NULL "
            "AND topic != '' ORDER BY topic"
        ).fetchall()
    return [r[0] for r in rows]


def get_doc(doc_id: int) -> dict | None:
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM docs WHERE id=?", (doc_id,)).fetchone()
    return dict(row) if row else None


def get_doc_by_slug(slug: str) -> dict | None:
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM docs WHERE slug=?", (slug,)).fetchone()
    return dict(row) if row else None


def create_doc(title: str, body: str, topic: str = "",
               pinned: bool = False, created_at: str | None = None) -> int:
    now = _now()
    with get_conn() as conn:
        slug = _unique_slug(conn, slugify(title))
        cur = conn.execute(
            "INSERT INTO docs (slug, title, topic, body, pinned, created_at, updated_at) "
            "VALUES (?,?,?,?,?,?,?)",
            (slug, title, topic or None, body, int(pinned),
             created_at or now, now),
        )
        return cur.lastrowid


def update_doc(doc_id: int, title: str, body: str, topic: str = "",
               pinned: bool = False) -> None:
    with get_conn() as conn:
        slug = _unique_slug(conn, slugify(title), exclude_id=doc_id)
        conn.execute(
            "UPDATE docs SET slug=?, title=?, topic=?, body=?, pinned=?, updated_at=? "
            "WHERE id=?",
            (slug, title, topic or None, body, int(pinned), _now(), doc_id),
        )


def delete_doc(doc_id: int) -> None:
    with get_conn() as conn:
        conn.execute("DELETE FROM docs WHERE id=?", (doc_id,))


def seed_docs() -> int:
    """Insert seed docs that are not already present (matched by title slug).

    Per-doc idempotent: docs whose slug already exists are skipped, so adding a
    new entry to ``seed_docs.SEED_DOCS`` lands it in existing databases on the
    next startup, while user edits to previously-seeded docs are never touched.

    Slugs listed in ``seed_docs.SUPERSEDED_DOC_SLUGS`` are deleted first. That
    is how a correction (e.g. a renumbering) propagates to an existing database:
    without it, the retired doc would sit alongside its replacement and the log
    would state two incompatible things at once.
    """
    # local import: seed data, not core schema
    from seed_docs import SEED_DOCS, SUPERSEDED_DOC_SLUGS

    # Retirements commit on their own connection before the insert loop opens
    # one. Holding a write lock across the loop would deadlock against the
    # separate connection create_doc() opens per document.
    if SUPERSEDED_DOC_SLUGS:
        with get_conn() as conn:
            conn.executemany(
                "DELETE FROM docs WHERE slug=?",
                [(slug,) for slug in SUPERSEDED_DOC_SLUGS],
            )

    added = 0
    with get_conn() as conn:
        for d in SEED_DOCS:
            slug = slugify(d["title"])
            if conn.execute(
                "SELECT id FROM docs WHERE slug=?", (slug,)
            ).fetchone():
                continue
            create_doc(
                title=d["title"],
                body=d["body"],
                topic=d.get("topic", ""),
                pinned=bool(d.get("pinned")),
                created_at=d.get("created_at"),
            )
            added += 1
    return added
