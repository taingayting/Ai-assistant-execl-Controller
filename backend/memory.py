"""Memory layer for Sara.

Stores contextual information between commands so users can issue follow-up
commands without repeating the filename or sheet name every time.

Stored state:
    active_file     – the Excel filename currently being worked on
    active_sheet    – the active sheet name within that file
    recent_commands – a bounded list of the last N executed actions

Persistence: backed by SQLite so state survives server restarts.
Thread-safety: all access is gated through a single ``threading.Lock``.
"""
import json
import logging
import os
import sqlite3
import threading
import time
from collections import deque

log = logging.getLogger('memory')

MAX_RECENT_COMMANDS = 20

_DB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
os.makedirs(_DB_DIR, exist_ok=True)
_DB_PATH = os.path.join(_DB_DIR, 'memory.db')

_SCHEMA = """
CREATE TABLE IF NOT EXISTS memory_state (
    key        TEXT PRIMARY KEY,
    value      TEXT,
    updated_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS recent_commands (
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    action  TEXT NOT NULL,
    ts      REAL NOT NULL
);
"""


class Memory:
    """Persistent context store for the Sara assistant (SQLite-backed).

    The ``memory_state`` table holds key/value pairs (``active_file``,
    ``active_sheet``) with an ``updated_at`` timestamp.  The
    ``recent_commands`` table keeps a bounded log of executed actions.

    In-memory caches mirror the DB for fast lock-free reads on the hot
    path, while every write is flushed to SQLite immediately so data
    survives server restarts.
    """

    def __init__(self, max_recent: int = MAX_RECENT_COMMANDS, db_path: str = _DB_PATH):
        self._lock = threading.Lock()
        self._max_recent = max_recent
        self._db_path = db_path
        self._local = threading.local()
        self._ensure_schema()
        self._migrate_legacy_kv()
        # Load cached copies for fast reads
        self._active_file: str | None = self._db_get('active_file')
        self._active_sheet: str | None = self._db_get('active_sheet')
        # Column headers of the active sheet — used to give the LLM context
        # for follow-up commands (e.g. "add another row").
        raw_headers = self._db_get('active_headers')
        self._active_headers: list[str] = json.loads(raw_headers) if raw_headers else []
        self._recent_commands: deque = deque(maxlen=max_recent)
        self._load_recent_commands()

    # ---- SQLite helpers ----

    def _get_conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, 'conn', None)
        if conn is None:
            conn = sqlite3.connect(self._db_path, timeout=10)
            conn.execute("PRAGMA journal_mode=WAL")
            self._local.conn = conn
        return conn

    def _ensure_schema(self):
        conn = self._get_conn()
        conn.executescript(_SCHEMA)
        conn.commit()

    def _migrate_legacy_kv(self):
        """One-time migration: copy rows from the old ``kv`` table (if it
        exists) into ``memory_state``, then drop ``kv``."""
        conn = self._get_conn()
        cur = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='kv'"
        )
        if cur.fetchone() is None:
            return  # nothing to migrate
        try:
            now = time.time()
            for row in conn.execute("SELECT key, value FROM kv"):
                conn.execute(
                    "INSERT OR IGNORE INTO memory_state (key, value, updated_at) "
                    "VALUES (?, ?, ?)",
                    (row[0], row[1], now),
                )
            conn.execute("DROP TABLE kv")
            conn.commit()
            log.info('memory: migrated legacy kv table -> memory_state')
        except Exception:
            log.warning('memory: legacy kv migration failed', exc_info=True)

    def _db_get(self, key: str) -> str | None:
        conn = self._get_conn()
        cur = conn.execute(
            "SELECT value FROM memory_state WHERE key = ?", (key,)
        )
        row = cur.fetchone()
        return row[0] if row else None

    def _db_set(self, key: str, value: str | None):
        conn = self._get_conn()
        now = time.time()
        if value is None:
            conn.execute("DELETE FROM memory_state WHERE key = ?", (key,))
        else:
            conn.execute(
                "INSERT INTO memory_state (key, value, updated_at) "
                "VALUES (?, ?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value, "
                "updated_at=excluded.updated_at",
                (key, value, now),
            )
        conn.commit()

    def _load_recent_commands(self):
        conn = self._get_conn()
        cur = conn.execute(
            "SELECT action FROM recent_commands ORDER BY id DESC LIMIT ?",
            (self._max_recent,),
        )
        rows = cur.fetchall()
        self._recent_commands.clear()
        for (raw,) in reversed(rows):
            try:
                self._recent_commands.append(json.loads(raw))
            except Exception:
                pass

    # ---- active_headers ----

    def get_active_headers(self) -> list[str]:
        """Return the column headers of the currently active sheet (may be empty)."""
        with self._lock:
            return list(self._active_headers)

    def set_active_headers(self, headers: list[str] | None) -> None:
        """Store the column headers for the active sheet.

        Pass an empty list or None to clear.
        """
        safe = list(headers) if headers else []
        with self._lock:
            self._active_headers = safe
            self._db_set('active_headers', json.dumps(safe, ensure_ascii=False))
        log.info('active_headers updated: %s', safe)

    # ---- active_file ----

    def get_active_file(self) -> str | None:
        with self._lock:
            return self._active_file

    def set_active_file(self, filename: str | None):
        with self._lock:
            prev = self._active_file
            self._active_file = filename
            self._db_set('active_file', filename)
        log.info('active_file: %s -> %s', prev, filename)

    # ---- active_sheet ----

    def get_active_sheet(self) -> str | None:
        with self._lock:
            return self._active_sheet

    def set_active_sheet(self, sheet: str | None):
        with self._lock:
            prev = self._active_sheet
            self._active_sheet = sheet
            self._db_set('active_sheet', sheet)
        log.info('active_sheet: %s -> %s', prev, sheet)

    # ---- recent_commands ----

    def add_command(self, action: dict):
        with self._lock:
            self._recent_commands.append(action)
            now = time.time()
            conn = self._get_conn()
            conn.execute(
                "INSERT INTO recent_commands (action, ts) VALUES (?, ?)",
                (json.dumps(action, ensure_ascii=False), now),
            )
            conn.commit()
            # Trim old rows in DB
            conn.execute(
                "DELETE FROM recent_commands WHERE id NOT IN "
                "(SELECT id FROM recent_commands ORDER BY id DESC LIMIT ?)",
                (self._max_recent,),
            )
            conn.commit()
            # Also update memory_state with serialised recent_commands
            self._db_set(
                'recent_commands',
                json.dumps(list(self._recent_commands), ensure_ascii=False),
            )
        log.debug('recent_commands: appended %s (len=%d)',
                  action.get('type', '?'), len(self._recent_commands))

    def get_recent_commands(self, n: int | None = None) -> list[dict]:
        with self._lock:
            items = list(self._recent_commands)
        if n is not None:
            return items[-n:]
        return items

    # ---- bulk access / reset ----

    def snapshot(self) -> dict:
        """Return a JSON-serialisable copy of the full memory state."""
        with self._lock:
            return {
                "active_file": self._active_file,
                "active_sheet": self._active_sheet,
                "active_headers": list(self._active_headers),
                "recent_commands": list(self._recent_commands),
            }

    def clear(self):
        """Reset all stored state."""
        with self._lock:
            self._active_file = None
            self._active_sheet = None
            self._active_headers = []
            self._recent_commands.clear()
            conn = self._get_conn()
            conn.execute("DELETE FROM memory_state")
            conn.execute("DELETE FROM recent_commands")
            conn.commit()
        log.info('memory cleared')


# Module-level singleton used by app.py and other modules.
memory = Memory()
