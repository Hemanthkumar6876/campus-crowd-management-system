"""SQLite storage for the historical crowd database and the alert log."""
import sqlite3
from pathlib import Path

import pandas as pd

from config import LOCATIONS

DB_DIR = Path(__file__).parent / "data"
DB_PATH = DB_DIR / "crowd.db"


def _conn():
    DB_DIR.mkdir(exist_ok=True)
    return sqlite3.connect(DB_PATH)


def init_db():
    with _conn() as c:
        c.execute(
            """CREATE TABLE IF NOT EXISTS observations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT NOT NULL,
                location TEXT NOT NULL,
                count INTEGER NOT NULL,
                source TEXT DEFAULT 'upload')"""
        )
        c.execute("CREATE INDEX IF NOT EXISTS ix_obs ON observations(location, ts)")
        c.execute(
            """CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT NOT NULL,
                location TEXT NOT NULL,
                level TEXT NOT NULL,
                count INTEGER,
                occupancy REAL,
                kind TEXT,
                message TEXT,
                acknowledged INTEGER DEFAULT 0)"""
        )


def count_rows():
    with _conn() as c:
        return c.execute("SELECT COUNT(*) FROM observations").fetchone()[0]


def version():
    """Cheap fingerprint used to invalidate Streamlit caches."""
    with _conn() as c:
        return c.execute("SELECT COALESCE(MAX(id),0), COUNT(*) FROM observations").fetchone()


def add_observations(rows):
    """rows: iterable of (ts, location, count, source); ts is datetime or str."""
    clean = []
    for ts, loc, cnt, src in rows:
        if loc not in LOCATIONS:
            continue
        clean.append((pd.Timestamp(ts).strftime("%Y-%m-%d %H:%M:%S"), loc, int(cnt), src))
    with _conn() as c:
        c.executemany(
            "INSERT INTO observations (ts, location, count, source) VALUES (?,?,?,?)", clean
        )
    return len(clean)


def load_observations(start=None, end=None, locations=None):
    q, args = "SELECT id, ts, location, count, source FROM observations WHERE 1=1", []
    if start is not None:
        q += " AND ts >= ?"
        args.append(pd.Timestamp(start).strftime("%Y-%m-%d %H:%M:%S"))
    if end is not None:
        q += " AND ts <= ?"
        args.append(pd.Timestamp(end).strftime("%Y-%m-%d %H:%M:%S"))
    if locations:
        q += " AND location IN (%s)" % ",".join("?" * len(locations))
        args += list(locations)
    with _conn() as c:
        df = pd.read_sql_query(q + " ORDER BY ts", c, params=args)
    df["ts"] = pd.to_datetime(df["ts"])
    return df


def clear_observations():
    with _conn() as c:
        c.execute("DELETE FROM observations")


def add_alert(ts, location, level, count, occupancy, kind, message):
    with _conn() as c:
        c.execute(
            "INSERT INTO alerts (ts, location, level, count, occupancy, kind, message) "
            "VALUES (?,?,?,?,?,?,?)",
            (pd.Timestamp(ts).strftime("%Y-%m-%d %H:%M:%S"), location, level, int(count),
             float(occupancy), kind, message),
        )


def load_alerts():
    with _conn() as c:
        df = pd.read_sql_query("SELECT * FROM alerts ORDER BY ts DESC, id DESC", c)
    if not df.empty:
        df["ts"] = pd.to_datetime(df["ts"])
    return df


def acknowledge_alerts(ids=None):
    with _conn() as c:
        if ids is None:
            c.execute("UPDATE alerts SET acknowledged=1")
        else:
            c.executemany("UPDATE alerts SET acknowledged=1 WHERE id=?", [(int(i),) for i in ids])


def clear_alerts():
    with _conn() as c:
        c.execute("DELETE FROM alerts")
