"""Tiny SQLite store for analysis history (standard library only)."""
import os
import sqlite3
import threading
from datetime import datetime

_lock = threading.Lock()


class History:
    def __init__(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.path = path
        with self._conn() as c:
            c.execute("""CREATE TABLE IF NOT EXISTS analyses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created TEXT, filename TEXT, label TEXT, forged INTEGER,
                score REAL, confidence REAL, regions INTEGER, seconds REAL, folder TEXT)""")

    def _conn(self):
        c = sqlite3.connect(self.path)
        c.row_factory = sqlite3.Row
        return c

    def add(self, filename, label, forged, score, confidence, regions, seconds, folder):
        with _lock, self._conn() as c:
            cur = c.execute(
                "INSERT INTO analyses (created, filename, label, forged, score, confidence, regions, seconds, folder) "
                "VALUES (?,?,?,?,?,?,?,?,?)",
                (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), filename, label, int(forged), score, confidence,
                 regions, seconds, folder))
            return cur.lastrowid

    def list(self, limit=50):
        with self._conn() as c:
            return [dict(r) for r in c.execute("SELECT * FROM analyses ORDER BY id DESC LIMIT ?", (limit,))]

    def get(self, rid):
        with self._conn() as c:
            r = c.execute("SELECT * FROM analyses WHERE id=?", (rid,)).fetchone()
            return dict(r) if r else None

    def delete(self, rid):
        with _lock, self._conn() as c:
            c.execute("DELETE FROM analyses WHERE id=?", (rid,))

    def stats(self):
        with self._conn() as c:
            r = c.execute("SELECT COUNT(*) n, COALESCE(SUM(forged),0) f FROM analyses").fetchone()
            return {"total": r["n"], "forged": r["f"]}
