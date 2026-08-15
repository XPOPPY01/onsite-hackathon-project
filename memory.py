"""In-memory store. The default, and the reference implementation.

The engine is a local batch tool: a run of a few hundred thousand events fits in
memory comfortably, and requiring a database file to process a CSV would be
friction with no benefit. :class:`~zoro_engine.store.SQLiteStore` exists for when
durability or SQL access is actually wanted.

This class defines the semantics the SQLite store must match - the round-trip
test in ``tests/test_store.py`` runs the same assertions against both.
"""

from __future__ import annotations

from typing import Any, Iterable, Iterator

from ..errors import StoreError
from ..models import Decision, Event
from .codec import transaction_to_row


class MemoryStore:
    """Events, decisions and state held in process."""

    __slots__ = ("_decisions", "_events", "_runs", "_seen", "_transactions")

    def __init__(self) -> None:
        self._events: list[Event] = []
        self._decisions: list[Decision] = []
        #: event_id -> seq. A dict rather than a set so a duplicate report can
        #: name the arrival position of the original.
        self._seen: dict[str, int] = {}
        self._transactions: list[dict[str, Any]] = []
        self._runs: list[dict[str, Any]] = []

    # --- duplicate ledger --------------------------------------------------

    def has_event(self, event_id: str) -> bool:
        return event_id in self._seen

    def seen_ids(self) -> frozenset[str]:
        return frozenset(self._seen)

    def seq_of(self, event_id: str) -> int | None:
        """Arrival position of an already-accepted event, if any."""
        return self._seen.get(event_id)

    # --- event log ---------------------------------------------------------

    def add_event(self, event: Event) -> None:
        if event.event_id in self._seen:
            raise StoreError(
                f"event {event.event_id} is already stored",
                event_id=event.event_id,
                existing_seq=self._seen[event.event_id],
            )
        self._seen[event.event_id] = event.seq
        self._events.append(event)

    def events(self) -> Iterator[Event]:
        # Sorted by seq rather than trusting insertion order, so this holds even
        # if events were loaded from a store that returned them unordered.
        yield from sorted(self._events, key=lambda item: item.seq)

    def events_for_user(self, user_key: str) -> Iterator[Event]:
        for event in self.events():
            if event.user_key == user_key:
                yield event

    def event_count(self) -> int:
        return len(self._events)

    def next_seq(self) -> int:
        return len(self._events) + 1

    # --- decisions ---------------------------------------------------------

    def add_decision(self, decision: Decision) -> None:
        self._decisions.append(decision)

    def decisions(self) -> Iterator[Decision]:
        yield from sorted(self._decisions, key=lambda item: item.seq)

    def decision_count(self) -> int:
        return len(self._decisions)

    # --- materialized state ------------------------------------------------

    def save_transactions(self, transactions: Iterable[Any]) -> None:
        self._transactions = [transaction_to_row(txn) for txn in transactions]

    def transaction_rows(self) -> list[dict[str, Any]]:
        return list(self._transactions)

    def save_run(self, record: dict[str, Any]) -> None:
        self._runs.append(dict(record))

    def runs(self) -> list[dict[str, Any]]:
        return list(self._runs)

    # --- lifecycle ---------------------------------------------------------

    def commit(self) -> None:
        """No-op: there is nothing to flush."""

    def close(self) -> None:
        """No-op: kept so callers can treat both stores identically."""

    def __enter__(self) -> MemoryStore:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def __repr__(self) -> str:  # pragma: no cover - display only
        return (
            f"MemoryStore(events={len(self._events)}, "
            f"decisions={len(self._decisions)})"
        )
