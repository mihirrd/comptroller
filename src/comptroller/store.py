import sqlite3
import time
from pathlib import Path

# Resolved DB paths we've already checked this process (avoids PRAGMA on every connect).
_schema_checked: set[str] = set()


def _migrate_sessions_schema(conn: sqlite3.Connection) -> None:
    cursor = conn.cursor()
    cursor.execute("PRAGMA table_info(sessions)")
    cols = {row[1] for row in cursor.fetchall()}
    if "api_dollars_used" not in cols:
        cursor.execute("ALTER TABLE sessions ADD COLUMN api_dollars_used REAL NOT NULL DEFAULT 0")
    if "max_api_dollars" not in cols:
        cursor.execute("ALTER TABLE sessions ADD COLUMN max_api_dollars REAL")
    if "wall_seconds_used" not in cols:
        cursor.execute("ALTER TABLE sessions ADD COLUMN wall_seconds_used REAL NOT NULL DEFAULT 0")
    if "max_wall_seconds" not in cols:
        cursor.execute("ALTER TABLE sessions ADD COLUMN max_wall_seconds REAL")
    if "session_retries_used" not in cols:
        cursor.execute("ALTER TABLE sessions ADD COLUMN session_retries_used INTEGER NOT NULL DEFAULT 0")
    if "max_session_retries" not in cols:
        cursor.execute("ALTER TABLE sessions ADD COLUMN max_session_retries INTEGER")
    if "primary_model" not in cols:
        cursor.execute("ALTER TABLE sessions ADD COLUMN primary_model TEXT")
    if "active_model" not in cols:
        cursor.execute("ALTER TABLE sessions ADD COLUMN active_model TEXT")
    if "model_degraded" not in cols:
        cursor.execute("ALTER TABLE sessions ADD COLUMN model_degraded INTEGER NOT NULL DEFAULT 0")
    if "local_model_url" not in cols:
        cursor.execute("ALTER TABLE sessions ADD COLUMN local_model_url TEXT")
    if "local_model_id" not in cols:
        cursor.execute("ALTER TABLE sessions ADD COLUMN local_model_id TEXT")
    conn.commit()


def _ensure_sessions_schema(conn: sqlite3.Connection, db_path_resolved: str) -> None:
    """Run additive migrations once per DB file per process.

    Migration used to run on every ``connect()`` so old databases picked up new columns
    even if ``init`` was never re-run. After the first successful check, we skip
    ``PRAGMA table_info`` for that path until the process exits.
    """
    if db_path_resolved in _schema_checked:
        return
    _migrate_sessions_schema(conn)
    _schema_checked.add(db_path_resolved)


def _db_key(db_path: str) -> str:
    return str(Path(db_path).expanduser().resolve())


def _connect(db_path: str) -> sqlite3.Connection:
    raw = str(Path(db_path).expanduser())
    conn = sqlite3.connect(raw)
    _ensure_sessions_schema(conn, _db_key(raw))
    return conn


def init_db(db_path: str):
    """Initialize database with required tables."""
    raw = str(Path(db_path).expanduser())
    Path(raw).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(raw)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            session_id  TEXT PRIMARY KEY,
            task        TEXT,
            status      TEXT,
            tokens_used INTEGER,
            max_tokens  INTEGER,
            summary     TEXT,
            created_at  REAL,
            updated_at  REAL
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS steps (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id  TEXT,
            step        INTEGER,
            event_type  TEXT,
            content     TEXT,
            tokens      INTEGER,
            timestamp   REAL
        )
    """)

    conn.commit()
    _ensure_sessions_schema(conn, _db_key(raw))
    conn.close()


def create_session(
    session_id: str,
    task: str,
    max_tokens: int,
    db_path: str = "~/.agent-runtime-mvp/sessions.db",
    *,
    max_api_dollars: float | None = None,
    max_wall_seconds: float | None = None,
    max_session_retries: int | None = None,
    primary_model: str | None = None,
    active_model: str | None = None,
    local_model_url: str | None = None,
    local_model_id: str | None = None,
):
    """Create a new session."""
    db_path = str(Path(db_path).expanduser())
    conn = _connect(db_path)
    cursor = conn.cursor()

    now = time.time()
    pm = primary_model or active_model
    am = active_model or primary_model
    cursor.execute("""
        INSERT INTO sessions (
            session_id, task, status, tokens_used, max_tokens,
            api_dollars_used, max_api_dollars,
            wall_seconds_used, max_wall_seconds,
            session_retries_used, max_session_retries,
            summary, created_at, updated_at,
            primary_model, active_model, model_degraded,
            local_model_url, local_model_id
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        session_id, task, "running", 0, max_tokens,
        0.0, max_api_dollars,
        0.0, max_wall_seconds,
        0, max_session_retries,
        None, now, now,
        pm, am, 0,
        local_model_url, local_model_id,
    ))

    conn.commit()
    conn.close()


def log_step(session_id: str, step: int, event_type: str, content: str, tokens: int,
             db_path: str = "~/.agent-runtime-mvp/sessions.db"):
    """Log a step in the session."""
    db_path = str(Path(db_path).expanduser())
    conn = _connect(db_path)
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO steps (session_id, step, event_type, content, tokens, timestamp)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (session_id, step, event_type, content, tokens, time.time()))

    conn.commit()
    conn.close()


def update_session(
    session_id: str,
    status: str,
    tokens_used: int,
    summary: str | None,
    api_dollars_used: float,
    wall_seconds_used: float,
    session_retries_used: int,
    db_path: str = "~/.agent-runtime-mvp/sessions.db",
    *,
    active_model: str | None = None,
    model_degraded: bool = False,
    local_model_url: str | None = None,
    local_model_id: str | None = None,
):
    """Update session status, token usage, dollar spend, wall time, retry count, and summary."""
    db_path = str(Path(db_path).expanduser())
    conn = _connect(db_path)
    cursor = conn.cursor()

    cursor.execute("""
        UPDATE sessions
        SET status = ?, tokens_used = ?, api_dollars_used = ?, wall_seconds_used = ?,
            session_retries_used = ?, summary = ?, updated_at = ?,
            active_model = ?, model_degraded = ?, local_model_url = ?, local_model_id = ?
        WHERE session_id = ?
    """, (
        status, tokens_used, api_dollars_used, wall_seconds_used,
        session_retries_used, summary, time.time(),
        active_model, 1 if model_degraded else 0, local_model_url, local_model_id,
        session_id,
    ))

    conn.commit()
    conn.close()


def update_session_budget_caps(
    session_id: str,
    db_path: str,
    *,
    max_tokens: int,
    max_api_dollars: float | None,
    max_wall_seconds: float | None,
):
    """Persist raised session budget caps (token, optional dollar, optional wall)."""
    db_path = str(Path(db_path).expanduser())
    conn = _connect(db_path)
    cursor = conn.cursor()
    cursor.execute(
        """
        UPDATE sessions
        SET max_tokens = ?, max_api_dollars = ?, max_wall_seconds = ?, updated_at = ?
        WHERE session_id = ?
        """,
        (max_tokens, max_api_dollars, max_wall_seconds, time.time(), session_id),
    )
    conn.commit()
    conn.close()


def get_session(session_id: str, db_path: str = "~/.agent-runtime-mvp/sessions.db") -> dict:
    """Get session by ID."""
    db_path = str(Path(db_path).expanduser())
    conn = _connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM sessions WHERE session_id = ?", (session_id,))
    row = cursor.fetchone()
    conn.close()

    if row:
        return dict(row)
    return {}


def get_steps(session_id: str, db_path: str = "~/.agent-runtime-mvp/sessions.db") -> list[dict]:
    """Get all steps for a session."""
    db_path = str(Path(db_path).expanduser())
    conn = _connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM steps WHERE session_id = ? ORDER BY step", (session_id,))
    rows = cursor.fetchall()
    conn.close()

    return [dict(row) for row in rows]


def list_sessions(db_path: str = "~/.agent-runtime-mvp/sessions.db") -> list[dict]:
    """List all sessions."""
    db_path = str(Path(db_path).expanduser())
    conn = _connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM sessions ORDER BY created_at DESC")
    rows = cursor.fetchall()
    conn.close()

    return [dict(row) for row in rows]
