"""Persistence for events, decisions and materialized state.

Two implementations behind one protocol:

* :class:`MemoryStore` - the default. The engine is a local batch tool and a run
  fits comfortably in memory, so persistence should not be mandatory.
* :class:`SQLiteStore` - durable, and the route to the PRD's SQL and Power BI
  deliverables. A relational schema rather than a JSON blob, so the analytics
  views are real SQL.

The protocol's central method is :meth:`Store.has_event`, the duplicate ledger
behind idempotency and the API's 409. It is the store's job rather than the
engine's because durability is exactly what the guarantee needs: an in-process
set would forget every event id the moment the process exited, and re-running an
import would double-count.
"""

from __future__ import annotations

from typing import Any, Iterable, Iterator, Protocol, Sequence, runtime_checkable

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
from .memory import MemoryStore
from .sqlite_store import SQLiteStore

__all__ = [
    "DECISION_COLUMNS",
    "EVENT_COLUMNS",
    "MemoryStore",
    "SQLiteStore",
    "Store",
    "TRANSACTION_COLUMNS",
    "decision_to_row",
    "event_to_row",
    "open_store",
    "row_to_decision",
    "row_to_event",
    "transaction_to_row",
]


@runtime_checkable
class Store(Protocol):
    """What the engine requires of a persistence layer."""

    # --- the duplicate ledger (R1 / HTTP 409) ------------------------------

    def has_event(self, event_id: str) -> bool:
        """True if this ``event_id`` was already accepted."""
        ...

    def seen_ids(self) -> frozenset[str]:
        """Every accepted ``event_id``. Used to warm an in-process cache."""
        ...

    # --- the append-only event log ----------------------------------------

    def add_event(self, event: Event) -> None:
        """Append an accepted event. Callers must check :meth:`has_event` first."""
        ...

    def events(self) -> Iterator[Event]:
        """Every event in **arrival order** - the sequence replay reproduces."""
        ...

    def events_for_user(self, user_key: str) -> Iterator[Event]:
        ...

    def event_count(self) -> int:
        ...

    def next_seq(self) -> int:
        """The sequence number for the next event."""
        ...

    # --- decisions ---------------------------------------------------------

    def add_decision(self, decision: Decision) -> None:
        ...

    def decisions(self) -> Iterator[Decision]:
        ...

    def decision_count(self) -> int:
        ...

    # --- materialized state ------------------------------------------------

    def save_transactions(self, transactions: Iterable[Any]) -> None:
        """Replace the reporting table with the current reconciled state."""
        ...

    def save_run(self, record: dict[str, Any]) -> None:
        """Record wall-clock run metadata (kept out of the hashed audit log)."""
        ...

    # --- lifecycle ---------------------------------------------------------

    def commit(self) -> None:
        ...

    def close(self) -> None:
        ...


def open_store(target: str | None = None, **kwargs: Any) -> Store:
    """Open a store from a location string.

    ``None``, ``":memory:"`` and ``"memory"`` give a :class:`MemoryStore`;
    anything else is treated as a SQLite path. Lets the CLI accept
    ``--store out/zoro.db`` and ``--store memory`` through one flag.
    """
    if target in (None, "", "memory", ":memory:"):
        return MemoryStore()
    return SQLiteStore(target, **kwargs)
