import sqlite3
from contextlib import contextmanager
from pathlib import Path

DB_PATH = Path(__file__).parent / "alerts.db"


def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS alerts (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker      TEXT    NOT NULL,
                above_price REAL,
                below_price REAL,
                email       TEXT    NOT NULL,
                active      INTEGER NOT NULL DEFAULT 1,
                created_at  TEXT    NOT NULL DEFAULT (datetime('now'))
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS alert_state (
                alert_id    INTEGER PRIMARY KEY REFERENCES alerts(id),
                above_fired INTEGER NOT NULL DEFAULT 0,
                below_fired INTEGER NOT NULL DEFAULT 0
            )
        """)


@contextmanager
def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def add_alert(ticker, above_price, below_price, email):
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO alerts (ticker, above_price, below_price, email) VALUES (?, ?, ?, ?)",
            (ticker.upper().strip(), above_price or None, below_price or None, email.strip()),
        )
        conn.execute("INSERT INTO alert_state (alert_id) VALUES (?)", (cur.lastrowid,))


def get_active_alerts():
    with get_conn() as conn:
        return [
            dict(r)
            for r in conn.execute(
                "SELECT id, ticker, above_price, below_price, email, created_at "
                "FROM alerts WHERE active = 1 ORDER BY created_at DESC"
            ).fetchall()
        ]


def deactivate_alert(alert_id):
    with get_conn() as conn:
        conn.execute("UPDATE alerts SET active = 0 WHERE id = ?", (alert_id,))


def get_alert_state(alert_id):
    with get_conn() as conn:
        row = conn.execute(
            "SELECT above_fired, below_fired FROM alert_state WHERE alert_id = ?",
            (alert_id,),
        ).fetchone()
    if row:
        return {"above_fired": bool(row["above_fired"]), "below_fired": bool(row["below_fired"])}
    return {"above_fired": False, "below_fired": False}


def update_alert_state(alert_id, above_fired, below_fired):
    with get_conn() as conn:
        conn.execute(
            "UPDATE alert_state SET above_fired = ?, below_fired = ? WHERE alert_id = ?",
            (int(above_fired), int(below_fired), alert_id),
        )
