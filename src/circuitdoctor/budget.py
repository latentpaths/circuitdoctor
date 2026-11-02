"""Transactional, durable reservations. Unsettled work remains charged at its estimate."""

from __future__ import annotations

import sqlite3
import time
import uuid
from pathlib import Path

STAGES = {"pilot": 40.0, "reproduction": 160.0, "experiments": 150.0, "adapters": 50.0, "reserve": 100.0}


class BudgetExceeded(RuntimeError):
    pass


class Ledger:
    def __init__(self, path: Path, cap: float = 500):
        if not 0 < cap <= 500:
            raise ValueError("Project cap must be in (0, 500]")
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path, self.cap = path, cap
        with self.connect() as db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS entries "
                "(id TEXT PRIMARY KEY, stage TEXT, label TEXT, reserved REAL, actual REAL, created REAL)"
            )
            db.execute(
                "CREATE TABLE IF NOT EXISTS counters "
                "(run TEXT, name TEXT, value INTEGER, PRIMARY KEY(run,name))"
            )

    def connect(self):
        db = sqlite3.connect(self.path, timeout=30)
        db.execute("PRAGMA busy_timeout=30000")
        return db

    def reserve(self, stage: str, label: str, amount: float) -> str:
        if stage not in STAGES or amount < 0:
            raise ValueError("Invalid stage or amount")
        key = uuid.uuid4().hex
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            total = db.execute("SELECT COALESCE(SUM(COALESCE(actual,reserved)),0) FROM entries").fetchone()[0]
            used = db.execute(
                "SELECT COALESCE(SUM(COALESCE(actual,reserved)),0) FROM entries WHERE stage=?", (stage,)
            ).fetchone()[0]
            if total + amount > self.cap + 1e-9 or used + amount > STAGES[stage] + 1e-9:
                raise BudgetExceeded(
                    f"Cannot reserve ${amount:.4f}: project ${total:.4f}/{self.cap}, "
                    f"{stage} ${used:.4f}/{STAGES[stage]}"
                )
            db.execute(
                "INSERT INTO entries VALUES (?,?,?,?,NULL,?)", (key, stage, label, amount, time.time())
            )
        return key

    def settle(self, key: str, actual: float):
        if actual < 0:
            raise ValueError("Negative cost")
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT reserved,actual FROM entries WHERE id=?", (key,)).fetchone()
            if row is None or row[1] is not None:
                raise ValueError("Unknown or already settled reservation")
            # Overspends are recorded honestly, then prevent further dispatch.
            db.execute("UPDATE entries SET actual=? WHERE id=?", (actual, key))
        if actual > row[0] + 1e-9:
            raise BudgetExceeded("Actual cost exceeded reservation; recorded, dispatch stopped")

    def consume(self, run: str, name: str, limit: int, count: int = 1):
        if count < 0:
            raise ValueError("Negative count")
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT value FROM counters WHERE run=? AND name=?", (run, name)).fetchone()
            value = (row[0] if row else 0) + count
            if value > limit:
                raise BudgetExceeded(f"{name} budget exhausted: {value}>{limit}")
            db.execute("INSERT OR REPLACE INTO counters VALUES (?,?,?)", (run, name, value))

    def summary(self):
        with self.connect() as db:
            rows = db.execute(
                "SELECT stage,SUM(COALESCE(actual,reserved)),SUM(actual IS NULL) FROM entries GROUP BY stage"
            ).fetchall()
            counters = db.execute("SELECT run,name,value FROM counters").fetchall()
        return {
            "cap_usd": self.cap,
            "committed_usd": sum(r[1] for r in rows),
            "stages": {r[0]: {"usd": r[1], "pending": r[2]} for r in rows},
            "counters": [{"run": r, "name": n, "value": v} for r, n, v in counters],
        }

    def snapshot(self, destination: Path):
        destination.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as source, sqlite3.connect(destination) as target:
            source.backup(target)

    def merge(self, other: Path):
        """Reconcile remote records by UUID, preserving settled local lease charges."""
        with sqlite3.connect(other) as remote:
            entries = remote.execute("SELECT * FROM entries").fetchall()
            counters = remote.execute("SELECT * FROM counters").fetchall()
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            for row in entries:
                current = db.execute("SELECT * FROM entries WHERE id=?", (row[0],)).fetchone()
                if current is None:
                    db.execute("INSERT INTO entries VALUES (?,?,?,?,?,?)", row)
                else:
                    if current[:4] != row[:4] or current[5] != row[5]:
                        raise ValueError("Conflicting reservation identity")
                    if current[4] is not None and row[4] is not None and abs(current[4] - row[4]) > 1e-9:
                        raise ValueError("Conflicting settled charges")
                    if current[4] is None and row[4] is not None:
                        db.execute("UPDATE entries SET actual=? WHERE id=?", (row[4], row[0]))
            for run, name, value in counters:
                db.execute(
                    "INSERT INTO counters VALUES (?,?,?) ON CONFLICT(run,name) DO UPDATE "
                    "SET value=MAX(value,excluded.value)",
                    (run, name, value),
                )
        if self.summary()["committed_usd"] > self.cap:
            raise BudgetExceeded("Reconciled spend exceeds project cap; further dispatch must stop")
