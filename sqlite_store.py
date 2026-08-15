"""SQLite-backed store: durable persistence and the route to SQL / Power BI.

Chosen over a JSON file because the PRD asks for SQL and a Power BI bonus, and
both want a real relational schema (see ``schema.sql``). SQLite is in the standard
library, needs no server, and the PRD forbids distributed systems - so a single
local file is not a compromise here, it is the correct shape.

Two deliberate choices worth stating:

**The duplicate ledger is a UNIQUE constraint, not application logic.** Idempotency
that lives only in Python is idempotency that a crash mid-batch can lose. Here a
re-inserted ``event_id`` fails at the database level.

**Money never touches a REAL column.** Amounts are INTEGER minor units end to end.
Binary floating point cannot represent 0.10 exactly, and a cent lost to
representation error would corrupt the audit trail invisibly.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Iterable, Iterator

from ..errors import StoreError
from ..models import Decision, Event
from .codec import (
    DECISION_COLUMNS,
    EVENT_COLUMNS,
    TRANSACTION_COLUMNS,
    decision_to_row,
    event_to_row,
    row_to_decision,
    row_to_event,
    transaction_to_row,
)

SCHEMA_PATH = Path(__file__).with_name("schema.sql")


def _insert_sql(table: str, columns: tuple[str, ...]) -> str:
    placeholders = ", ".join(f":{name}" for name in columns)
    # Quoted because several column names (date, action, source) are either
    # reserved words or close enough to be worth not gambling on.
    quoted = ", ".join(f'"{name}"' for name in columns)
    return f'INSERT INTO "{table}" ({quoted}) VALUES ({placeholders})'


_INSERT_EVENT = _insert_sql("events", EVENT_COLUMNS)
_INSERT_DECISION = _insert_sql("decisions", DECISION_COLUMNS)
_INSERT_TRANSACTION = _insert_sql("transactions", TRANSACTION_COLUMNS)

#: Ordering for replay. Explicitly by arrival ``seq``: reproducing a run means
#: replaying the same arrival sequence, and the fold re-sorts into timeline order
#: internally.
_SELECT_EVENTS = f'SELECT {", ".join(EVENT_COLUMNS)} FROM events ORDER BY seq'


class SQLiteStore:
    """Durable store over a single SQLite file."""

    __slots__ = ("_conn", "_seen", "path")

    def __init__(
        self,
        path: str | Path = ":memory:",
        *,
        timeout: float = 30.0,
    ) -> None:
        self.path = Path(path) if str(path) != ":memory:" else Path(":memory:")
        if str(path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)

        try:
            self._conn = sqlite3.connect(str(path), timeout=timeout)
        except sqlite3.Error as exc:
            raise StoreError(f"could not open {path}: {exc}", path=str(path)) from None

        self._conn.row_factory = sqlite3.Row
        # Enforced per-connection in SQLite, not stored with the schema.
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._apply_schema()

        #: In-process mirror of the UNIQUE index. The constraint remains the real
        #: guarantee; this only avoids a query per incoming event.
        self._seen: set[str] = {
            str(row[0]) for row in self._conn.execute("SELECT event_id FROM events")
        }

    def _apply_schema(self) -> None:
        try:
            self._conn.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
        except OSError as exc:
            raise StoreError(f"could not read schema: {exc}", path=str(SCHEMA_PATH)) from None
        except sqlite3.Error as exc:
            raise StoreError(f"could not apply schema: {exc}", path=str(self.path)) from None
        self._conn.commit()

    # --- duplicate ledger --------------------------------------------------

    def has_event(self, event_id: str) -> bool:
        return event_id in self._seen

    def seen_ids(self) -> frozenset[str]:
        return frozenset(self._seen)

    def seq_of(self, event_id: str) -> int | None:
        row = self._conn.execute(
            "SELECT seq FROM events WHERE event_id = ?", (event_id,)
        ).fetchone()
        return int(row["seq"]) if row is not None else None

    # --- event log ---------------------------------------------------------

    def add_event(self, event: Event) -> None:
        try:
            self._conn.execute(_INSERT_EVENT, event_to_row(event))
        except sqlite3.IntegrityError as exc:
            # IntegrityError covers UNIQUE *and* NOT NULL/CHECK. Reporting them
            # all as "already stored" would send someone hunting a duplicate that
            # does not exist while the real defect is a malformed column.
            if "UNIQUE" in str(exc):
                raise StoreError(
                    f"event {event.event_id} is already stored",
                    event_id=event.event_id,
                ) from None
            raise StoreError(
                f"event {event.event_id} violates the schema: {exc}",
                event_id=event.event_id,
            ) from None
        except sqlite3.Error as exc:
            raise StoreError(f"could not store event: {exc}", event_id=event.event_id) from None
        self._seen.add(event.event_id)

    def add_events(self, events: Iterable[Event]) -> None:
        """Batch insert - one transaction instead of one per row."""
        rows = [event_to_row(event) for event in events]
        if not rows:
            return
        try:
            self._conn.executemany(_INSERT_EVENT, rows)
        except sqlite3.IntegrityError as exc:
            raise StoreError(f"duplicate event in batch: {exc}") from None
        self._seen.update(str(row["event_id"]) for row in rows)

    def events(self) -> Iterator[Event]:
        for row in self._conn.execute(_SELECT_EVENTS):
            yield row_to_event(row)

    def events_for_user(self, user_key: str) -> Iterator[Event]:
        sql = (
            f'SELECT {", ".join(EVENT_COLUMNS)} FROM events '
            "WHERE user_key = ? ORDER BY seq"
        )
        for row in self._conn.execute(sql, (user_key,)):
            yield row_to_event(row)

    def event_count(self) -> int:
        return int(self._conn.execute("SELECT COUNT(*) FROM events").fetchone()[0])

    def next_seq(self) -> int:
        row = self._conn.execute("SELECT MAX(seq) FROM events").fetchone()
        return int(row[0] or 0) + 1

    # --- decisions ---------------------------------------------------------

    def add_decision(self, decision: Decision) -> None:
        try:
            self._conn.execute(_INSERT_DECISION, decision_to_row(decision))
        except sqlite3.Error as exc:
            raise StoreError(
                f"could not store decision: {exc}", event_id=decision.event_id
            ) from None

    def decisions(self) -> Iterator[Decision]:
        sql = f'SELECT {", ".join(DECISION_COLUMNS)} FROM decisions ORDER BY seq'
        for row in self._conn.execute(sql):
            yield row_to_decision(row)

    def decision_count(self) -> int:
        return int(self._conn.execute("SELECT COUNT(*) FROM decisions").fetchone()[0])

    # --- materialized state ------------------------------------------------

    def save_transactions(self, transactions: Iterable[Any]) -> None:
        """Replace the reporting table wholesale.

        DELETE-then-INSERT rather than UPSERT: a transaction is the *output* of
        folding its event log, so patching rows in place is exactly how
        materialized state drifts away from the events that produced it. Wrapped
        in one transaction so a reader never observes an empty table.
        """
        rows = [transaction_to_row(txn) for txn in transactions]
        try:
            with self._conn:
                self._conn.execute("DELETE FROM transactions")
                if rows:
                    self._conn.executemany(_INSERT_TRANSACTION, rows)
        except sqlite3.Error as exc:
            raise StoreError(f"could not save transactions: {exc}") from None

    def transaction_rows(self) -> list[dict[str, Any]]:
        sql = (
            f'SELECT {", ".join(TRANSACTION_COLUMNS)} FROM transactions '
            "ORDER BY user_id, transaction_key"
        )
        return [dict(row) for row in self._conn.execute(sql)]

    def save_run(self, record: dict[str, Any]) -> None:
        columns = tuple(record)
        quoted = ", ".join(f'"{name}"' for name in columns)
        placeholders = ", ".join(f":{name}" for name in columns)
        try:
            self._conn.execute(
                f"INSERT OR REPLACE INTO runs ({quoted}) VALUES ({placeholders})", record
            )
        except sqlite3.Error as exc:
            raise StoreError(f"could not save run metadata: {exc}") from None

    def runs(self) -> list[dict[str, Any]]:
        return [
            dict(row)
            for row in self._conn.execute("SELECT * FROM runs ORDER BY started_at, run_id")
        ]

    # --- SQL access --------------------------------------------------------

    def query(self, sql: str, params: Iterable[Any] = ()) -> list[dict[str, Any]]:
        """Run a read-only query. Backs the analytics views and CLI reporting."""
        try:
            return [dict(row) for row in self._conn.execute(sql, tuple(params))]
        except sqlite3.Error as exc:
            raise StoreError(f"query failed: {exc}", sql=sql) from None

    def executescript(self, sql: str) -> None:
        """Apply a SQL script, such as ``sql/analytics_views.sql``."""
        try:
            self._conn.executescript(sql)
            self._conn.commit()
        except sqlite3.Error as exc:
            raise StoreError(f"script failed: {exc}") from None

    # --- lifecycle ---------------------------------------------------------

    def commit(self) -> None:
        self._conn.commit()

    def close(self) -> None:
        try:
            self._conn.commit()
        finally:
            self._conn.close()

    def __enter__(self) -> SQLiteStore:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def __repr__(self) -> str:  # pragma: no cover - display only
        return f"SQLiteStore(path={str(self.path)!r}, events={self.event_count()})"
