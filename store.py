import sqlite3
import time
from pathlib import Path


def init_db(db_path: str):
    """Initialize database with required tables."""
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
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
    conn.close()


def create_session(session_id: str, task: str, max_tokens: int, db_path: str = "~/.agent-runtime-mvp/sessions.db"):
    """Create a new session."""
    db_path = str(Path(db_path).expanduser())
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    now = time.time()
    cursor.execute("""
        INSERT INTO sessions (session_id, task, status, tokens_used, max_tokens, summary, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (session_id, task, "running", 0, max_tokens, None, now, now))

    conn.commit()
    conn.close()


def log_step(session_id: str, step: int, event_type: str, content: str, tokens: int,
             db_path: str = "~/.agent-runtime-mvp/sessions.db"):
    """Log a step in the session."""
    db_path = str(Path(db_path).expanduser())
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO steps (session_id, step, event_type, content, tokens, timestamp)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (session_id, step, event_type, content, tokens, time.time()))

    conn.commit()
    conn.close()


def update_session(session_id: str, status: str, tokens_used: int, summary: str | None,
                   db_path: str = "~/.agent-runtime-mvp/sessions.db"):
    """Update session status and summary."""
    db_path = str(Path(db_path).expanduser())
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("""
        UPDATE sessions
        SET status = ?, tokens_used = ?, summary = ?, updated_at = ?
        WHERE session_id = ?
    """, (status, tokens_used, summary, time.time(), session_id))

    conn.commit()
    conn.close()


def get_session(session_id: str, db_path: str = "~/.agent-runtime-mvp/sessions.db") -> dict:
    """Get session by ID."""
    db_path = str(Path(db_path).expanduser())
    conn = sqlite3.connect(db_path)
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
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM steps WHERE session_id = ? ORDER BY step", (session_id,))
    rows = cursor.fetchall()
    conn.close()

    return [dict(row) for row in rows]


def list_sessions(db_path: str = "~/.agent-runtime-mvp/sessions.db") -> list[dict]:
    """List all sessions."""
    db_path = str(Path(db_path).expanduser())
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM sessions ORDER BY created_at DESC")
    rows = cursor.fetchall()
    conn.close()

    return [dict(row) for row in rows]
