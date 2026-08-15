"""Core data structures: Event, Transaction, Provenance, Decision.

Three ideas carry most of the engine's correctness and are worth reading before
the rest of the code:

**1. ``Event.supplied``** - the set of fields the input row actually asserted.
A partial update (``category`` only) must not null out ``amount``, so the fold
never reads a field the event did not supply. Without this, "merge" degrades
into "overwrite with mostly-blanks".

**2. ``Provenance.precedence``** - every materialized field remembers *who* set
it and with what authority. Conflict resolution compares precedence tuples
rather than arrival order, which is what lets a late-arriving bank sync lose to
an earlier user correction.

**3. ``Event.canonical_key``** - the total order used to rebuild a transaction's
timeline. It contains no arrival-order component, so the final state is
independent of the order events reached the system.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import date as Date
from datetime import datetime
from typing import Any, Iterable, Mapping

from .canonical import format_ts, sha256_hex
from .config import ACTION_RANK, MUTABLE_FIELDS, TEXT_FIELDS
from .money import format_amount

# Fields that carry structural links rather than financial values. Kept out of
# MUTABLE_FIELDS because an explicitly-supplied blank *does* unlink, whereas a
# blank category never clears a known one.
STRUCTURAL_FIELDS: tuple[str, ...] = ("parent_transaction_id",)

#: Every field the resolver may write, in canonical (hash-stable) order.
RESOLVED_FIELDS: tuple[str, ...] = MUTABLE_FIELDS + STRUCTURAL_FIELDS


# ---------------------------------------------------------------------------
# Normalization issues
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class Issue:
    """One problem found while normalizing a row.

    ``severity="error"`` rejects the row; ``"warning"`` accepts it but records
    what we changed, so a rounded amount or a synthesized id is never silent.
    """

    code: str
    message: str
    field: str = ""
    severity: str = "error"

    @property
    def is_error(self) -> bool:
        return self.severity == "error"

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "field": self.field,
            "message": self.message,
            "severity": self.severity,
        }

    def __str__(self) -> str:  # pragma: no cover - display only
        where = f" ({self.field})" if self.field else ""
        return f"{self.code}{where}: {self.message}"


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class Provenance:
    """Who asserted a field value, and with what authority.

    The ordering of :attr:`precedence` encodes the engine's central policy
    decision: **authority outranks recency**. A newer low-authority event loses
    to an older high-authority one, which is the only reading of the PRD's
    "prefer user edits over bank syncs" that survives out-of-order delivery.
    """

    source: str
    authority: int
    event_ts: datetime
    action: str
    event_id: str

    @property
    def precedence(self) -> tuple[int, datetime, int, str]:
        """Comparable tuple; strictly greater wins a field contest.

        ``event_id`` is the final tie-break. It is content-derived, so the
        winner is arbitrary but *stable* - and the resolver flags such ties as
        conflicts rather than hiding them.
        """
        return (self.authority, self.event_ts, ACTION_RANK.get(self.action, 1), self.event_id)

    def beats(self, other: Provenance | None) -> bool:
        """True if this assertion outranks ``other`` (``None`` = unset field)."""
        if other is None:
            return True
        return self.precedence > other.precedence

    def ties_with(self, other: Provenance | None) -> bool:
        """True on equal authority *and* equal timestamp - a genuine conflict."""
        if other is None:
            return False
        return self.authority == other.authority and self.event_ts == other.event_ts

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "authority": self.authority,
            "event_ts": format_ts(self.event_ts),
            "action": self.action,
            "event_id": self.event_id,
        }


# ---------------------------------------------------------------------------
# Event
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class Event:
    """A normalized, validated financial event. Immutable by construction.

    Produced only by :mod:`zoro_engine.normalize`; the resolver never sees a raw
    row. ``amount_minor`` is integer minor units and ``event_ts`` is always
    timezone-aware UTC, so every comparison downstream is exact.
    """

    # --- identity ---------------------------------------------------------
    event_id: str
    user_id: str          # display form, first-seen casing
    user_key: str         # identity form used for grouping
    transaction_id: str   # display form
    transaction_key: str  # identity form

    # --- intent -----------------------------------------------------------
    action: str           # create | update | delete
    source: str           # canonical source
    event_ts: datetime    # UTC; when the change was asserted

    # --- payload ----------------------------------------------------------
    currency: str
    amount_minor: int | None = None
    date: Date | None = None
    category: str | None = None
    merchant: str | None = None
    account_id: str | None = None
    parent_transaction_id: str | None = None

    # --- resolution inputs ------------------------------------------------
    #: Fields this event actually asserted. The fold reads nothing else.
    supplied: frozenset[str] = frozenset()
    role: str = "standard"
    authority: int = 0

    # --- provenance / diagnostics ----------------------------------------
    seq: int = 0                      # arrival order, assigned at ingest
    origin: str = "api"               # "csv:<file>:<line>" | "api" | "replay"
    source_raw: str = ""              # source string as supplied
    warnings: tuple[str, ...] = ()
    flags: tuple[str, ...] = ()       # id_synthesized, ts_inferred, ...
    raw: Mapping[str, Any] = field(default_factory=dict)

    # --- ordering ---------------------------------------------------------

    @property
    def canonical_key(self) -> tuple[datetime, int, str]:
        """Total order for rebuilding a transaction's timeline.

        Contains **no arrival-order term** on purpose: the same set of events
        folds to the same state no matter what order they arrived in. That is
        what makes late and out-of-order delivery a non-event for the engine.
        """
        return (self.event_ts, ACTION_RANK.get(self.action, 1), self.event_id)

    @property
    def provenance(self) -> Provenance:
        return Provenance(
            source=self.source,
            authority=self.authority,
            event_ts=self.event_ts,
            action=self.action,
            event_id=self.event_id,
        )

    @property
    def txn_ref(self) -> tuple[str, str]:
        """The key this event reconciles against."""
        return (self.user_key, self.transaction_key)

    def asserts(self, name: str) -> bool:
        """True if this event supplied ``name`` (and so may change it)."""
        return name in self.supplied

    def value_of(self, name: str) -> Any:
        return getattr(self, name, None)

    def with_seq(self, seq: int) -> Event:
        """Copy stamped with its arrival sequence."""
        return replace(self, seq=seq)

    # --- serialization ----------------------------------------------------

    def to_dict(self, *, include_raw: bool = False) -> dict[str, Any]:
        """Canonical, JSON-safe view. Amount rendered as an exact string."""
        payload: dict[str, Any] = {
            "event_id": self.event_id,
            "user_id": self.user_id,
            "transaction_id": self.transaction_id,
            "action": self.action,
            "source": self.source,
            "event_ts": format_ts(self.event_ts),
            "amount": format_amount(self.amount_minor, self.currency),
            "amount_minor": self.amount_minor,
            "currency": self.currency,
            "date": self.date.isoformat() if self.date else None,
            "category": self.category,
            "merchant": self.merchant,
            "account_id": self.account_id,
            "parent_transaction_id": self.parent_transaction_id,
            "supplied": sorted(self.supplied),
            "role": self.role,
            "authority": self.authority,
            "seq": self.seq,
            "origin": self.origin,
        }
        if self.warnings:
            payload["warnings"] = list(self.warnings)
        if self.flags:
            payload["flags"] = list(self.flags)
        if include_raw:
            payload["raw"] = {str(k): (None if v is None else str(v)) for k, v in self.raw.items()}
        return payload

    def to_evidence(self) -> dict[str, Any]:
        """Compact form for the ``evidence`` list in a decision record.

        Trimmed to the fields that justify a resolution outcome, so audit
        records stay readable when a transaction has a long history.
        """
        item: dict[str, Any] = {
            "event_id": self.event_id,
            "action": self.action,
            "source": self.source,
            "authority": self.authority,
            "event_ts": format_ts(self.event_ts),
            "seq": self.seq,
        }
        if self.asserts("amount_minor"):
            item["amount"] = format_amount(self.amount_minor, self.currency)
        if self.asserts("date") and self.date:
            item["date"] = self.date.isoformat()
        if self.flags:
            item["flags"] = list(self.flags)
        return item


# ---------------------------------------------------------------------------
# Transaction
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class Transaction:
    """The materialized state of one transaction, plus its field provenance.

    Rebuilt from scratch by folding the transaction's event log on every new
    event (see :mod:`zoro_engine.resolution`). Rebuilding rather than mutating
    is what makes a late event correct instead of merely detected.
    """

    user_id: str
    user_key: str
    transaction_id: str
    transaction_key: str
    currency: str

    amount_minor: int | None = None
    date: Date | None = None
    category: str | None = None
    merchant: str | None = None
    account_id: str | None = None
    parent_transaction_id: str | None = None

    #: Tombstoned. The row is retained after a delete so that a late create or
    #: update reconciles against it instead of silently resurrecting the row.
    deleted: bool = False
    #: Materialized from an update whose create has not arrived yet (R6).
    provisional: bool = False

    version: int = 0
    event_count: int = 0
    provenance: Mapping[str, Provenance] = field(default_factory=dict)

    # --- derived ----------------------------------------------------------

    @property
    def exists(self) -> bool:
        """Visible to downstream financial models."""
        return not self.deleted

    @property
    def amount(self) -> str | None:
        return format_amount(self.amount_minor, self.currency)

    def provenance_of(self, name: str) -> Provenance | None:
        return self.provenance.get(name)

    # --- serialization ----------------------------------------------------

    def snapshot(self) -> dict[str, Any]:
        """Financial state only - the view downstream models consume.

        Excludes provenance and version so that ``state_diff`` reports what
        actually changed about the *money*, not bookkeeping churn.
        """
        return {
            "transaction_id": self.transaction_id,
            "amount": format_amount(self.amount_minor, self.currency),
            "amount_minor": self.amount_minor,
            "currency": self.currency,
            "date": self.date.isoformat() if self.date else None,
            "category": self.category,
            "merchant": self.merchant,
            "account_id": self.account_id,
            "parent_transaction_id": self.parent_transaction_id,
            "deleted": self.deleted,
            "provisional": self.provisional,
        }

    def to_public(self) -> dict[str, Any]:
        """Snapshot plus bookkeeping, for API and ``final_state.json``."""
        payload = self.snapshot()
        payload["version"] = self.version
        payload["event_count"] = self.event_count
        return payload

    def to_full(self) -> dict[str, Any]:
        """Snapshot plus per-field provenance, for the authority hash."""
        payload = self.snapshot()
        payload["provenance"] = {
            name: prov.to_dict() for name, prov in sorted(self.provenance.items())
        }
        return payload

    def state_hash(self) -> str:
        return sha256_hex(self.snapshot())


# ---------------------------------------------------------------------------
# State diffing
# ---------------------------------------------------------------------------

#: Diff-only keys, excluded from per-field comparison.
_DIFF_SKIP: frozenset[str] = frozenset({"transaction_id"})


def diff_snapshots(
    before: Mapping[str, Any] | None,
    after: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Build the ``state_diff`` payload for an audit record.

    Shape::

        {"op": "created|updated|deleted|restored|none",
         "fields": {"amount": {"from": "12.00", "to": "15.00"}, ...}}

    ``op`` names the lifecycle transition and ``fields`` the value-level change,
    because a reviewer needs both: "deleted" without the field values does not
    explain what was lost, and field values without "deleted" do not explain why
    the row vanished from the totals.
    """
    if before is None and after is None:
        return {"op": "none", "fields": {}}

    fields: dict[str, dict[str, Any]] = {}
    keys = sorted((set(before or {}) | set(after or {})) - _DIFF_SKIP)
    for key in keys:
        old = (before or {}).get(key)
        new = (after or {}).get(key)
        if old != new:
            fields[key] = {"from": old, "to": new}

    if before is None:
        op = "created"
    elif after is None:
        op = "purged"
    elif not before.get("deleted") and after.get("deleted"):
        op = "deleted"
    elif before.get("deleted") and not after.get("deleted"):
        op = "restored"
    elif fields:
        op = "updated"
    else:
        op = "none"

    return {"op": op, "fields": fields}


def diff_is_empty(state_diff: Mapping[str, Any]) -> bool:
    """True when nothing about the financial state changed."""
    return state_diff.get("op") == "none" and not state_diff.get("fields")


# ---------------------------------------------------------------------------
# Decision (audit record)
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class Decision:
    """The engine's verdict on one event - the unit of the audit trail.

    ``timestamp`` is the *event's* timestamp, never a wall clock. That keeps the
    audit log byte-for-byte reproducible across runs, which is what makes
    ``zoro verify`` a meaningful check rather than a formality. Wall-clock run
    metadata lives in ``run_manifest.json`` instead.
    """

    seq: int
    event_id: str
    user_id: str
    transaction_id: str
    action: str
    source: str
    timestamp: datetime

    decision: str          # created | merged | replaced | deleted | ignored | rejected
    rule: str              # e.g. "R5_UPDATE_MERGE"
    reason: str            # machine-readable reason code + detail

    state_diff: dict[str, Any] = field(default_factory=lambda: {"op": "none", "fields": {}})
    evidence: tuple[dict[str, Any], ...] = ()
    conflicts: tuple[dict[str, Any], ...] = ()
    anomalies: tuple[dict[str, Any], ...] = ()
    issues: tuple[dict[str, Any], ...] = ()
    warnings: tuple[str, ...] = ()

    user_version: int = 0
    txn_version: int = 0
    state_hash: str = ""       # user state hash after applying this event
    prev_hash: str = ""        # previous record's hash (tamper-evident chain)
    record_hash: str = ""

    @property
    def changed_state(self) -> bool:
        return not diff_is_empty(self.state_diff)

    def body(self) -> dict[str, Any]:
        """The hashed payload: everything except the chain hashes themselves."""
        payload: dict[str, Any] = {
            "seq": self.seq,
            "timestamp": format_ts(self.timestamp),
            "event_id": self.event_id,
            "user_id": self.user_id,
            "transaction_id": self.transaction_id,
            "action": self.action,
            "source": self.source,
            "decision": self.decision,
            "rule": self.rule,
            "reason": self.reason,
            "state_diff": self.state_diff,
            "evidence": list(self.evidence),
            "user_version": self.user_version,
            "txn_version": self.txn_version,
            "state_hash": self.state_hash,
        }
        # Optional blocks are omitted when empty so the common record stays
        # small and greppable; hashes stay stable because absence is canonical.
        if self.conflicts:
            payload["conflicts"] = list(self.conflicts)
        if self.anomalies:
            payload["anomalies"] = list(self.anomalies)
        if self.issues:
            payload["issues"] = list(self.issues)
        if self.warnings:
            payload["warnings"] = list(self.warnings)
        return payload

    def to_audit_record(self) -> dict[str, Any]:
        """Full record as written to ``audit.log`` (one JSONL line)."""
        payload = self.body()
        payload["prev_hash"] = self.prev_hash
        payload["record_hash"] = self.record_hash
        return payload

    def to_response(self) -> dict[str, Any]:
        """API-facing view of the decision."""
        payload = self.body()
        payload["record_hash"] = self.record_hash
        return payload

    def comparable(self) -> dict[str, Any]:
        """Replay-comparison view.

        Excludes the chain hashes (which depend on preceding records) but keeps
        every field that a resolution rule can influence, so a replay mismatch
        points at the rule that drifted.
        """
        return self.body()


def build_decision(**kwargs: Any) -> Decision:
    """Construct a :class:`Decision`, normalizing sequence types to tuples."""
    for key in ("evidence", "conflicts", "anomalies", "issues", "warnings"):
        if key in kwargs and kwargs[key] is not None and not isinstance(kwargs[key], tuple):
            kwargs[key] = tuple(kwargs[key])
    return Decision(**kwargs)


def seal_decision(decision: Decision, prev_hash: str) -> Decision:
    """Attach the tamper-evident chain hashes to a finished decision."""
    from .canonical import chain_hash  # local import: avoids a cycle at module load

    return replace(
        decision,
        prev_hash=prev_hash,
        record_hash=chain_hash(prev_hash, decision.body()),
    )


def evidence_from(events: Iterable[Event]) -> tuple[dict[str, Any], ...]:
    """Compact evidence list from the events the resolver considered."""
    return tuple(event.to_evidence() for event in events)
