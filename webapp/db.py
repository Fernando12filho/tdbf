"""SQLite database layer for the TDBF webapp."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent / "tdbf.db"


def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


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

            CREATE INDEX IF NOT EXISTS idx_runs_instance ON runs(instance_id);
            CREATE INDEX IF NOT EXISTS idx_runs_model    ON runs(model);
            CREATE INDEX IF NOT EXISTS idx_runs_level    ON runs(best_level);
        """)


def insert_run(record: dict) -> int:
    flags = record.get("flags", {})
    attempts_data = record.get("attempts", [])

    with get_conn() as conn:
        cur = conn.execute(
            """
            INSERT INTO runs
                (instance_id, repo, model, split, think, context, observe,
                 best_level, best_attempt, total_attempts, timestamp)
            VALUES (?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                record.get("instance_id", ""),
                record.get("repo", ""),
                record.get("model", ""),
                record.get("split", "lite"),
                int(flags.get("think", False)),
                int(flags.get("context", False)),
                int(flags.get("observe", False)),
                record.get("best_level"),
                record.get("best_attempt"),
                len(attempts_data),
                record.get("timestamp", ""),
            ),
        )
        run_id = cur.lastrowid

        for a in attempts_data:
            conn.execute(
                """
                INSERT INTO attempts
                    (run_id, attempt_number, level, outcome_buggy, outcome_fixed,
                     test_code, probe_code, observed_output, raw_response,
                     prompt_sent, probe_prompt, input_tokens, output_tokens, duration_seconds)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    run_id,
                    a.get("attempt", 0),
                    a.get("level"),
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

def get_dashboard_stats() -> dict:
    with get_conn() as conn:
        total = conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
        by_level = conn.execute(
            "SELECT best_level, COUNT(*) as cnt FROM runs GROUP BY best_level ORDER BY best_level"
        ).fetchall()
        by_model = conn.execute(
            """
            SELECT model,
                   COUNT(*) as total,
                   SUM(CASE WHEN best_level=4 THEN 1 ELSE 0 END) as l4,
                   SUM(CASE WHEN best_level>=3 THEN 1 ELSE 0 END) as candidates
            FROM runs GROUP BY model ORDER BY model
            """
        ).fetchall()
    return {
        "total": total,
        "by_level": [dict(r) for r in by_level],
        "by_model": [dict(r) for r in by_model],
    }


def get_runs(model: str | None = None, level: int | None = None, limit: int = 100) -> list[dict]:
    clauses, params = [], []
    if model:
        clauses.append("model = ?")
        params.append(model)
    if level is not None:
        clauses.append("best_level = ?")
        params.append(level)
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
