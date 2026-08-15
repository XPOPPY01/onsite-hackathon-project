"""Serializing events and decisions for storage.

Shared by every :class:`~zoro_engine.store.Store` implementation on purpose. If
the in-memory store and the SQLite store disagreed about how an event
round-trips, replay from disk would diverge from replay from memory and the
divergence would look like a resolution bug rather than a storage bug.

The round-trip is **lossless for every field the fold reads**. That is the bar:
``row_to_event(event_to_row(e))`` must produce an event that folds identically,
which is verified by asserting the reconstructed ``canonical_key``,
``supplied`` set and ``authority`` all match.
"""

from __future__ import annotations

import json
from datetime import date as Date
from typing import Any, Mapping

from ..canonical import format_ts, parse_ts
from ..models import Decision, Event

#: Columns of the ``events`` table, in declaration order.
EVENT_COLUMNS = (
    "seq", "event_id", "user_id", "user_key", "transaction_id", "transaction_key",
    "action", "source", "source_raw", "event_ts", "currency", "amount_minor",
    "date", "category", "merchant", "account_id", "parent_transaction_id",
    "supplied", "role", "authority", "origin", "warnings", "flags", "raw",
)

#: Columns of the ``decisions`` table, in declaration order.
DECISION_COLUMNS = (
    "seq", "event_id", "user_id", "transaction_id", "action", "source", "timestamp",
    "decision", "rule", "reason", "state_diff", "evidence", "conflicts", "anomalies",
    "issues", "warnings", "user_version", "txn_version", "state_hash",
    "prev_hash", "record_hash",
)


def _dump(value: Any) -> str | None:
    """JSON-encode a container, or ``None`` when it is empty.

    Empty collections are stored as SQL ``NULL`` rather than ``"[]"`` so that
    ``WHERE warnings IS NOT NULL`` is a usable filter in the analytics views.

    Only for fields where empty genuinely means "nothing to report". Use
    :func:`_dump_always` where emptiness is itself a fact.
    """
    if not value:
        return None
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _dump_always(value: Any) -> str:
    """JSON-encode a container, emitting ``"[]"`` rather than ``NULL`` when empty.

    Required for ``supplied``: a **delete asserts no fields at all**, so an empty
    set is its correct and meaningful value, not missing data. Collapsing it to
    ``NULL`` makes every delete event unstorable against a ``NOT NULL`` column -
    and would make "asserted nothing" indistinguishable from "unknown" if the
    column were nullable.
    """
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _load(raw: Any, default: Any) -> Any:
    if raw in (None, ""):
        return default
    if isinstance(raw, (list, dict)):
        return raw
    try:
        return json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return default


def event_to_row(event: Event) -> dict[str, Any]:
    """Flatten an event into storable columns."""
    return {
        "seq": event.seq,
        "event_id": event.event_id,
        "user_id": event.user_id,
        "user_key": event.user_key,
        "transaction_id": event.transaction_id,
        "transaction_key": event.transaction_key,
        "action": event.action,
        "source": event.source,
        "source_raw": event.source_raw,
        "event_ts": format_ts(event.event_ts),
        "currency": event.currency,
        "amount_minor": event.amount_minor,
        "date": event.date.isoformat() if event.date else None,
        "category": event.category,
        "merchant": event.merchant,
        "account_id": event.account_id,
        "parent_transaction_id": event.parent_transaction_id,
        # Sorted so the stored bytes are stable for a given event. Always
        # written, never NULL: a delete supplies no fields, and that is a fact.
        "supplied": _dump_always(sorted(event.supplied)),
        "role": event.role,
        "authority": event.authority,
        "origin": event.origin,
        "warnings": _dump(list(event.warnings)),
        "flags": _dump(list(event.flags)),
        "raw": _dump({str(k): (None if v is None else str(v)) for k, v in event.raw.items()}),
    }


def row_to_event(row: Mapping[str, Any]) -> Event:
    """Rebuild an event from stored columns.

    Every field that participates in ordering or resolution is restored:
    ``event_ts`` back to aware UTC, ``date`` back to a real date, ``supplied``
    back to a frozenset. A string where a frozenset belongs would silently make
    ``asserts()`` true for every single character.
    """
    raw_date = row["date"]
    return Event(
        event_id=row["event_id"],
        user_id=row["user_id"],
        user_key=row["user_key"],
        transaction_id=row["transaction_id"],
        transaction_key=row["transaction_key"],
        action=row["action"],
        source=row["source"],
        event_ts=parse_ts(row["event_ts"]),
        currency=row["currency"],
        amount_minor=row["amount_minor"],
        date=Date.fromisoformat(raw_date) if raw_date else None,
        category=row["category"],
        merchant=row["merchant"],
        account_id=row["account_id"],
        parent_transaction_id=row["parent_transaction_id"],
        supplied=frozenset(_load(row["supplied"], [])),
        role=row["role"],
        authority=int(row["authority"]),
        seq=int(row["seq"]),
        origin=row["origin"],
        source_raw=row["source_raw"] or "",
        warnings=tuple(_load(row["warnings"], [])),
        flags=tuple(_load(row["flags"], [])),
        raw=_load(row["raw"], {}),
    )


def decision_to_row(decision: Decision) -> dict[str, Any]:
    """Flatten a decision into storable columns.

    The nested blocks (``state_diff``, ``evidence``, ...) are stored as JSON. The
    scalar columns are duplicated out of them so SQL and Power BI can group by
    ``decision`` or ``rule`` without parsing JSON.
    """
    return {
        "seq": decision.seq,
        "event_id": decision.event_id,
        "user_id": decision.user_id,
        "transaction_id": decision.transaction_id,
        "action": decision.action,
        "source": decision.source,
        "timestamp": format_ts(decision.timestamp),
        "decision": decision.decision,
        "rule": decision.rule,
        "reason": decision.reason,
        "state_diff": _dump(decision.state_diff),
        "evidence": _dump(list(decision.evidence)),
        "conflicts": _dump(list(decision.conflicts)),
        "anomalies": _dump(list(decision.anomalies)),
        "issues": _dump(list(decision.issues)),
        "warnings": _dump(list(decision.warnings)),
        "user_version": decision.user_version,
        "txn_version": decision.txn_version,
        "state_hash": decision.state_hash,
        "prev_hash": decision.prev_hash,
        "record_hash": decision.record_hash,
    }


def row_to_decision(row: Mapping[str, Any]) -> Decision:
    """Rebuild a decision from stored columns."""
    return Decision(
        seq=int(row["seq"]),
        event_id=row["event_id"],
        user_id=row["user_id"],
        transaction_id=row["transaction_id"],
        action=row["action"],
        source=row["source"],
        timestamp=parse_ts(row["timestamp"]),
        decision=row["decision"],
        rule=row["rule"],
        reason=row["reason"],
        state_diff=_load(row["state_diff"], {"op": "none", "fields": {}}),
        evidence=tuple(_load(row["evidence"], [])),
        conflicts=tuple(_load(row["conflicts"], [])),
        anomalies=tuple(_load(row["anomalies"], [])),
        issues=tuple(_load(row["issues"], [])),
        warnings=tuple(_load(row["warnings"], [])),
        user_version=int(row["user_version"] or 0),
        txn_version=int(row["txn_version"] or 0),
        state_hash=row["state_hash"] or "",
        prev_hash=row["prev_hash"] or "",
        record_hash=row["record_hash"] or "",
    )


#: Columns of the ``transactions`` table - the materialized state, denormalized
#: for SQL and Power BI. Rewritten wholesale on each save, because a transaction
#: is a fold output rather than an accumulating row.
TRANSACTION_COLUMNS = (
    "user_id", "transaction_id", "transaction_key", "currency", "amount_minor",
    "amount", "date", "month", "category", "merchant", "account_id",
    "parent_transaction_id", "deleted", "provisional", "version", "provenance",
)


def transaction_to_row(txn: Any) -> dict[str, Any]:
    """Flatten a materialized transaction for the reporting table."""
    from ..money import format_amount

    return {
        "user_id": txn.user_id,
        "transaction_id": txn.transaction_id,
        "transaction_key": txn.transaction_key,
        "currency": txn.currency,
        "amount_minor": txn.amount_minor,
        # Rendered alongside the integer so a report never re-derives it and
        # risks a different rounding convention.
        "amount": format_amount(txn.amount_minor, txn.currency),
        "date": txn.date.isoformat() if txn.date else None,
        "month": txn.date.strftime("%Y-%m") if txn.date else None,
        "category": txn.category,
        "merchant": txn.merchant,
        "account_id": txn.account_id,
        "parent_transaction_id": txn.parent_transaction_id,
        "deleted": 1 if txn.deleted else 0,
        "provisional": 1 if txn.provisional else 0,
        "version": txn.version,
        "provenance": _dump({
            name: prov.to_dict() for name, prov in sorted(txn.provenance.items())
        }),
    }
