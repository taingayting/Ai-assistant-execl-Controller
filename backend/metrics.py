"""Metrics collector for Sara.

Provides a thread-safe singleton that records every LLM routing decision,
command execution, and latency measurement.  Data is persisted to an SQLite
database so it survives restarts and can be queried for research evaluation.

Usage::

    from metrics import metrics
    metrics.record_llm_call(text, provider='groq', latency_ms=142, success=True)
    metrics.record_execution(action_type='add_row', success=True, latency_ms=23)
    summary = metrics.summary()
"""
from __future__ import annotations

import csv
import io
import json
import logging
import os
import sqlite3
import threading
import time
from contextlib import contextmanager

log = logging.getLogger('metrics')

_DB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
os.makedirs(_DB_DIR, exist_ok=True)
_DB_PATH = os.path.join(_DB_DIR, 'metrics.db')

_SCHEMA = """
CREATE TABLE IF NOT EXISTS pipeline_stages (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ts          REAL    NOT NULL,
    request_id  TEXT,
    session_id  TEXT,
    stage       TEXT    NOT NULL,
    latency_ms  REAL    NOT NULL DEFAULT 0,
    success     INTEGER NOT NULL DEFAULT 1,
    detail      TEXT
);

CREATE TABLE IF NOT EXISTS sessions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT    UNIQUE NOT NULL,
    started_at  REAL    NOT NULL,
    ended_at    REAL,
    commands    INTEGER NOT NULL DEFAULT 0,
    successes   INTEGER NOT NULL DEFAULT 0,
    wake_source TEXT
);

CREATE TABLE IF NOT EXISTS llm_calls (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ts          REAL    NOT NULL,
    request_id  TEXT,
    user_text   TEXT    NOT NULL,
    provider    TEXT    NOT NULL,
    model       TEXT,
    action_type TEXT,
    success     INTEGER NOT NULL DEFAULT 1,
    latency_ms  REAL    NOT NULL DEFAULT 0,
    fallback    INTEGER NOT NULL DEFAULT 0,
    raw_result  TEXT
);

CREATE TABLE IF NOT EXISTS executions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ts          REAL    NOT NULL,
    request_id  TEXT,
    action_type TEXT    NOT NULL,
    workflow_id TEXT,
    success     INTEGER NOT NULL DEFAULT 1,
    latency_ms  REAL    NOT NULL DEFAULT 0,
    error       TEXT,
    filename    TEXT
);

CREATE TABLE IF NOT EXISTS requests (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ts          REAL    NOT NULL,
    request_id  TEXT    NOT NULL,
    endpoint    TEXT    NOT NULL,
    method      TEXT    NOT NULL DEFAULT 'POST',
    status_code INTEGER,
    latency_ms  REAL    NOT NULL DEFAULT 0
);
"""


def _percentile(sorted_values: list[float], pct: float) -> float:
    """Return the *pct*-th percentile of a pre-sorted list (e.g. pct=95 → P95).

    Returns 0 for empty lists.  Uses nearest-rank method.
    """
    if not sorted_values:
        return 0.0
    k = max(0, int(len(sorted_values) * pct / 100) - 1)
    return sorted_values[min(k, len(sorted_values) - 1)]


class MetricsCollector:
    """Thread-safe metrics store backed by SQLite."""

    def __init__(self, db_path: str = _DB_PATH):
        self._db_path = db_path
        self._local = threading.local()
        self._init_lock = threading.Lock()
        self._ensure_schema()

    # ---- connection management ----

    def _get_conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, 'conn', None)
        if conn is None:
            conn = sqlite3.connect(self._db_path, timeout=10)
            conn.execute("PRAGMA journal_mode=WAL")
            self._local.conn = conn
        return conn

    @contextmanager
    def _cursor(self):
        conn = self._get_conn()
        cur = conn.cursor()
        try:
            yield cur
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    def _ensure_schema(self):
        with self._init_lock:
            conn = sqlite3.connect(self._db_path, timeout=10)
            try:
                conn.executescript(_SCHEMA)
                conn.commit()
            finally:
                conn.close()

    # ---- recording ----

    def record_llm_call(self, user_text: str, *, provider: str,
                        model: str | None = None,
                        action_type: str | None = None,
                        success: bool = True,
                        latency_ms: float = 0,
                        fallback: bool = False,
                        request_id: str | None = None,
                        raw_result: dict | None = None):
        raw = json.dumps(raw_result, ensure_ascii=False) if raw_result else None
        with self._cursor() as cur:
            cur.execute(
                "INSERT INTO llm_calls (ts, request_id, user_text, provider, model, "
                "action_type, success, latency_ms, fallback, raw_result) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (time.time(), request_id, user_text, provider, model,
                 action_type, int(success), latency_ms, int(fallback), raw),
            )

    def record_execution(self, action_type: str, *, success: bool = True,
                         latency_ms: float = 0, workflow_id: str | None = None,
                         error: str | None = None, filename: str | None = None,
                         request_id: str | None = None):
        with self._cursor() as cur:
            cur.execute(
                "INSERT INTO executions (ts, request_id, action_type, workflow_id, "
                "success, latency_ms, error, filename) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (time.time(), request_id, action_type, workflow_id,
                 int(success), latency_ms, error, filename),
            )

    def record_request(self, endpoint: str, method: str = 'POST',
                       status: int = 200, latency_ms: float = 0,
                       request_id: str | None = None):
        with self._cursor() as cur:
            cur.execute(
                "INSERT INTO requests (ts, request_id, endpoint, method, "
                "status_code, latency_ms) VALUES (?, ?, ?, ?, ?, ?)",
                (time.time(), request_id, endpoint, method,
                 status, latency_ms),
            )

    # ---- queries ----

    def summary(self) -> dict:
        """Return aggregate metrics suitable for /api/metrics."""
        conn = self._get_conn()
        cur = conn.cursor()

        # Total LLM calls
        cur.execute("SELECT COUNT(*) FROM llm_calls")
        total_llm = cur.fetchone()[0]

        # LLM success rate
        cur.execute("SELECT COUNT(*) FROM llm_calls WHERE success = 1")
        llm_success = cur.fetchone()[0]

        # Provider breakdown
        cur.execute(
            "SELECT provider, COUNT(*) FROM llm_calls GROUP BY provider"
        )
        provider_counts = dict(cur.fetchall())

        # Fallback frequency
        cur.execute("SELECT COUNT(*) FROM llm_calls WHERE fallback = 1")
        fallback_count = cur.fetchone()[0]

        # LLM latency stats (avg + P50/P95)
        cur.execute("SELECT AVG(latency_ms) FROM llm_calls WHERE latency_ms > 0")
        avg_llm_latency = cur.fetchone()[0] or 0
        cur.execute("SELECT latency_ms FROM llm_calls WHERE latency_ms > 0 ORDER BY latency_ms")
        llm_latencies = [r[0] for r in cur.fetchall()]
        p50_llm = _percentile(llm_latencies, 50)
        p95_llm = _percentile(llm_latencies, 95)

        # Total executions
        cur.execute("SELECT COUNT(*) FROM executions")
        total_exec = cur.fetchone()[0]

        # Execution success rate
        cur.execute("SELECT COUNT(*) FROM executions WHERE success = 1")
        exec_success = cur.fetchone()[0]

        # Execution latency stats (avg + P50/P95)
        cur.execute("SELECT AVG(latency_ms) FROM executions WHERE latency_ms > 0")
        avg_exec_latency = cur.fetchone()[0] or 0
        cur.execute("SELECT latency_ms FROM executions WHERE latency_ms > 0 ORDER BY latency_ms")
        exec_latencies = [r[0] for r in cur.fetchall()]
        p50_exec = _percentile(exec_latencies, 50)
        p95_exec = _percentile(exec_latencies, 95)

        # Error categorisation for execution failures
        cur.execute(
            "SELECT error, COUNT(*) FROM executions "
            "WHERE success=0 AND error IS NOT NULL GROUP BY error ORDER BY COUNT(*) DESC LIMIT 10"
        )
        top_errors = [{"error": r[0], "count": r[1]} for r in cur.fetchall()]

        # Action type breakdown
        cur.execute(
            "SELECT action_type, COUNT(*) FROM executions GROUP BY action_type"
        )
        action_counts = dict(cur.fetchall())

        # Total API requests
        cur.execute("SELECT COUNT(*) FROM requests")
        total_requests = cur.fetchone()[0]

        # Average request latency
        cur.execute("SELECT AVG(latency_ms) FROM requests WHERE latency_ms > 0")
        avg_request_latency = cur.fetchone()[0] or 0

        return {
            "llm": {
                "total_calls": total_llm,
                "success_count": llm_success,
                "success_rate": round(llm_success / total_llm, 4) if total_llm else 0,
                "fallback_count": fallback_count,
                "fallback_rate": round(fallback_count / total_llm, 4) if total_llm else 0,
                "avg_latency_ms": round(avg_llm_latency, 2),
                "p50_latency_ms": round(p50_llm, 2),
                "p95_latency_ms": round(p95_llm, 2),
                "provider_breakdown": provider_counts,
            },
            "execution": {
                "total_commands": total_exec,
                "success_count": exec_success,
                "success_rate": round(exec_success / total_exec, 4) if total_exec else 0,
                "avg_latency_ms": round(avg_exec_latency, 2),
                "p50_latency_ms": round(p50_exec, 2),
                "p95_latency_ms": round(p95_exec, 2),
                "action_type_breakdown": action_counts,
                "top_errors": top_errors,
            },
            "api": {
                "total_requests": total_requests,
                "avg_latency_ms": round(avg_request_latency, 2),
            },
        }

    # ---- pipeline stage recording ----

    def record_pipeline_stage(self, stage: str, *, latency_ms: float = 0,
                               success: bool = True, detail: str | None = None,
                               request_id: str | None = None,
                               session_id: str | None = None) -> None:
        """Record one named stage of the voice pipeline with its latency.

        *stage* should be one of: ``'stt'``, ``'llm'``, ``'execution'``, ``'tts'``.
        Use this to build per-stage latency charts for the thesis.
        """
        with self._cursor() as cur:
            cur.execute(
                "INSERT INTO pipeline_stages "
                "(ts, request_id, session_id, stage, latency_ms, success, detail) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (time.time(), request_id, session_id, stage,
                 latency_ms, int(success), detail),
            )

    # ---- session recording ----

    def start_session(self, session_id: str, wake_source: str | None = None) -> None:
        """Record the start of a new wake-word session."""
        with self._cursor() as cur:
            cur.execute(
                "INSERT OR IGNORE INTO sessions "
                "(session_id, started_at, wake_source) VALUES (?, ?, ?)",
                (session_id, time.time(), wake_source),
            )

    def end_session(self, session_id: str,
                    commands: int = 0, successes: int = 0) -> None:
        """Record the end of a session with total command counts."""
        with self._cursor() as cur:
            cur.execute(
                "UPDATE sessions SET ended_at=?, commands=?, successes=? "
                "WHERE session_id=?",
                (time.time(), commands, successes, session_id),
            )

    # ---- enhanced queries ----

    def pipeline_stage_summary(self) -> dict:
        """Return per-stage average and P95 latency — key thesis table."""
        conn = self._get_conn()
        cur = conn.cursor()
        cur.execute(
            "SELECT stage, COUNT(*), AVG(latency_ms), "
            "SUM(CASE WHEN success=1 THEN 1 ELSE 0 END) "
            "FROM pipeline_stages GROUP BY stage"
        )
        rows = cur.fetchall()
        result: dict = {}
        for stage, count, avg_ms, success_count in rows:
            # Compute P95 by fetching sorted latencies
            cur.execute(
                "SELECT latency_ms FROM pipeline_stages WHERE stage=? "
                "ORDER BY latency_ms",
                (stage,),
            )
            latencies = [r[0] for r in cur.fetchall()]
            p95 = _percentile(latencies, 95) if latencies else 0
            p50 = _percentile(latencies, 50) if latencies else 0
            result[stage] = {
                "count": count,
                "avg_latency_ms": round(avg_ms or 0, 2),
                "p50_latency_ms": round(p50, 2),
                "p95_latency_ms": round(p95, 2),
                "success_rate": round(success_count / count, 4) if count else 0,
            }
        return result

    def session_summary(self) -> dict:
        """Return aggregate session statistics."""
        conn = self._get_conn()
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM sessions")
        total = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM sessions WHERE ended_at IS NOT NULL")
        completed = cur.fetchone()[0]
        cur.execute(
            "SELECT AVG(ended_at - started_at) FROM sessions "
            "WHERE ended_at IS NOT NULL"
        )
        avg_duration = cur.fetchone()[0] or 0
        cur.execute(
            "SELECT AVG(commands) FROM sessions WHERE ended_at IS NOT NULL"
        )
        avg_commands = cur.fetchone()[0] or 0
        return {
            "total_sessions": total,
            "completed_sessions": completed,
            "avg_duration_seconds": round(avg_duration, 1),
            "avg_commands_per_session": round(avg_commands, 2),
        }

    def recent_llm_calls(self, limit: int = 50) -> list[dict]:
        conn = self._get_conn()
        cur = conn.cursor()
        cur.execute(
            "SELECT ts, request_id, user_text, provider, model, action_type, "
            "success, latency_ms, fallback FROM llm_calls "
            "ORDER BY id DESC LIMIT ?", (limit,)
        )
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]

    def export_csv(self, table: str) -> str:
        """Return the contents of *table* as a CSV string.

        *table* must be one of: ``'llm_calls'``, ``'executions'``,
        ``'requests'``, ``'pipeline_stages'``, ``'sessions'``.

        Useful for thesis data export — write the result to a ``.csv`` file
        and import into Excel / matplotlib / R for analysis.
        """
        allowed = {'llm_calls', 'executions', 'requests', 'pipeline_stages', 'sessions'}
        if table not in allowed:
            raise ValueError(f"table must be one of {sorted(allowed)}")
        conn = self._get_conn()
        cur = conn.cursor()
        cur.execute(f"SELECT * FROM {table} ORDER BY id")  # noqa: S608 — table name validated above
        cols = [d[0] for d in cur.description]
        rows = cur.fetchall()
        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(cols)
        writer.writerows(rows)
        return buf.getvalue()

    def reset(self):
        """Clear all metrics data (for testing)."""
        with self._cursor() as cur:
            cur.execute("DELETE FROM llm_calls")
            cur.execute("DELETE FROM executions")
            cur.execute("DELETE FROM requests")
            cur.execute("DELETE FROM pipeline_stages")
            cur.execute("DELETE FROM sessions")


# Module-level singleton
metrics = MetricsCollector()
