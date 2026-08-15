"""Conflict detection and resolution - the deterministic fold.

This module is the engine's core. It answers one question: *given every event
ever seen for one transaction, what is that transaction's true state?*

The answer is computed by **folding the transaction's whole event log in
canonical timestamp order, from scratch, on every new event**. That is more work
than mutating a row in place, and it is the reason late and out-of-order events
are correct rather than merely detected: an event that arrives late is simply
inserted at its rightful position and the timeline is rebuilt around it. The
fold has no arrival-order input, so the same set of events always produces the
same state - which is the PRD's determinism requirement, structurally rather
than by convention.

Two axes govern every contest:

* **Order** comes from ``Event.canonical_key`` - ``(event_ts, action_rank,
  event_id)``. This is the timeline.
* **Authority** comes from ``Provenance.precedence`` - ``(authority, event_ts,
  action_rank, event_id)``. This decides who wins a field.

Keeping them separate is what makes "prefer user edits over bank syncs" hold
under out-of-order delivery: a bank sync that arrives *after* a user correction
still loses to it, because authority dominates recency.

The numbered rules R1-R11 are specified in ``docs/RESOLUTION_RULES.md``; the
constants below are the single source of truth for their identifiers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date as Date
from typing import Any, Callable, Iterable, Mapping, Sequence

from .canonical import format_ts
from .config import EngineConfig, TEXT_FIELDS
from .models import (
    Event,
    Provenance,
    Transaction,
    diff_is_empty,
)
from .money import format_amount

# ---------------------------------------------------------------------------
# Rule identifiers
# ---------------------------------------------------------------------------

R1_DUPLICATE_EVENT = "R1_DUPLICATE_EVENT"
R2_SCHEMA_REJECT = "R2_SCHEMA_REJECT"
R3_CREATE_NEW = "R3_CREATE_NEW"
R4_CREATE_DUPLICATE = "R4_CREATE_DUPLICATE"
R5_UPDATE_MERGE = "R5_UPDATE_MERGE"
R6_UPDATE_BEFORE_CREATE = "R6_UPDATE_BEFORE_CREATE"
R7_DELETE_APPLIED = "R7_DELETE_APPLIED"
R8_DELETE_BLOCKED = "R8_DELETE_BLOCKED"
R9_DELETE_BEFORE_CREATE = "R9_DELETE_BEFORE_CREATE"
#: Cross-cutting. Surfaces as a ``late_event_reordered`` annotation on the
#: ``reason`` of whichever action rule fired, since a late event does not
#: replace the action's rule - it changes where in the timeline it lands.
R10_LATE_EVENT_REORDERED = "R10_LATE_EVENT_REORDERED"
R11_TIE_BROKEN = "R11_TIE_BROKEN"

#: Cap on how many events are embedded as evidence in one audit record. A
#: transaction with hundreds of edits would otherwise make the audit log
#: unreadable; the count is always reported so nothing is silently hidden.
MAX_EVIDENCE_EVENTS = 50


# ---------------------------------------------------------------------------
# Field groups
# ---------------------------------------------------------------------------

#: Fields resolved as a unit. ``amount_minor`` and ``currency`` *must* move
#: together: ``1500`` is 15.00 USD but 1500 JPY, so letting two different events
#: win the two halves would silently corrupt the stored value. Every other field
#: is independent.
FIELD_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("amount", ("amount_minor", "currency")),
    ("date", ("date",)),
    ("category", ("category",)),
    ("merchant", ("merchant",)),
    ("account_id", ("account_id",)),
    ("parent_transaction_id", ("parent_transaction_id",)),
)

#: Group name -> the field whose presence means "this group was asserted".
_GROUP_TRIGGER: dict[str, str] = {name: fields[0] for name, fields in FIELD_GROUPS}


# ---------------------------------------------------------------------------
# Dependency guards for deletes (R8)
# ---------------------------------------------------------------------------

#: Given ``(user_key, transaction_key)``, return the display ids of live
#: transactions that depend on it. Supplied by the state layer.
ChildLookup = Callable[[str, str], Sequence[str]]


def _no_children(_user_key: str, _txn_key: str) -> Sequence[str]:
    return ()


@dataclass(frozen=True, slots=True)
class DependencyContext:
    """Resolves the PRD's "remove transaction if no dependencies exist".

    Two concrete dependencies are enforced:

    * **Live children** - another transaction points at this one via
      ``parent_transaction_id`` (an account merge or a split). Deleting the
      parent would orphan them and silently change their rollups.
    * **A locked accounting period** - the transaction's date falls inside a
      closed period. Reports have already been filed against it, so the delete
      is refused and recorded rather than applied.
    """

    config: EngineConfig
    children: ChildLookup = _no_children

    def blockers(self, event: Event, txn_date: Date | None) -> tuple[dict[str, Any], ...]:
        found: list[dict[str, Any]] = []

        if not self.config.enforce_delete_dependencies:
            return ()

        kids = tuple(self.children(event.user_key, event.transaction_key))
        if kids:
            found.append({
                "type": "live_children",
                "detail": f"{len(kids)} live transaction(s) reference this record as parent",
                "transaction_ids": list(kids[:20]),
                "count": len(kids),
            })

        period = self.config.is_locked(event.user_key, txn_date)
        if period is not None:
            found.append({
                "type": "locked_period",
                "detail": f"date {txn_date} falls in locked period "
                          f"{period.start.isoformat()}..{period.end.isoformat()}",
                "label": period.label,
            })

        return tuple(found)


# ---------------------------------------------------------------------------
# Fold results
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class EventEffect:
    """What one event actually achieved inside the fold.

    ``won``/``lost`` are the field groups this event's values do and do not
    currently hold. They are what make an audit record explanatory rather than
    merely declarative: "merged" plus ``won=['amount']``, ``lost=['date']`` says
    precisely which half of a conflicting edit survived.
    """

    event_id: str
    won: tuple[str, ...] = ()
    lost: tuple[str, ...] = ()
    blocked: str = ""
    lifecycle: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()

    @property
    def applied(self) -> bool:
        return not self.blocked


class _MutableEffect:
    """Accumulator for an :class:`EventEffect` during the fold."""

    __slots__ = ("event_id", "won", "lost", "blocked", "lifecycle", "notes")

    def __init__(self, event_id: str) -> None:
        self.event_id = event_id
        self.won: list[str] = []
        self.lost: list[str] = []
        self.blocked: str = ""
        self.lifecycle: list[str] = []
        self.notes: list[str] = []

    def freeze(self) -> EventEffect:
        return EventEffect(
            event_id=self.event_id,
            # Sorted so audit payloads are hash-stable regardless of the order
            # field groups happened to be evaluated in.
            won=tuple(sorted(self.won)),
            lost=tuple(sorted(self.lost)),
            blocked=self.blocked,
            lifecycle=tuple(self.lifecycle),
            notes=tuple(self.notes),
        )


@dataclass(frozen=True, slots=True)
class FoldResult:
    """Materialized outcome of folding one transaction's event log."""

    values: Mapping[str, Any]
    provenance: Mapping[str, Provenance]
    exists: bool
    deleted: bool
    provisional: bool
    effects: Mapping[str, EventEffect]
    conflicts: tuple[dict[str, Any], ...] = ()
    event_count: int = 0

    def build(
        self,
        *,
        user_id: str,
        user_key: str,
        transaction_id: str,
        transaction_key: str,
        version: int,
        default_currency: str,
    ) -> Transaction | None:
        """Assemble a :class:`Transaction`, or ``None`` if it never existed."""
        if not self.exists:
            return None
        return Transaction(
            user_id=user_id,
            user_key=user_key,
            transaction_id=transaction_id,
            transaction_key=transaction_key,
            currency=self.values.get("currency") or default_currency,
            amount_minor=self.values.get("amount_minor"),
            date=self.values.get("date"),
            category=self.values.get("category"),
            merchant=self.values.get("merchant"),
            account_id=self.values.get("account_id"),
            parent_transaction_id=self.values.get("parent_transaction_id"),
            deleted=self.deleted,
            provisional=self.provisional,
            version=version,
            event_count=self.event_count,
            provenance=dict(self.provenance),
        )

    def effect_for(self, event_id: str) -> EventEffect:
        return self.effects.get(event_id, EventEffect(event_id=event_id))


# ---------------------------------------------------------------------------
# The fold
# ---------------------------------------------------------------------------

def sort_events(events: Iterable[Event]) -> list[Event]:
    """Order events into their canonical timeline.

    Sorting by ``canonical_key`` - which excludes arrival order entirely - is
    what makes the fold arrival-order independent.
    """
    return sorted(events, key=lambda event: event.canonical_key)


def fold_transaction(
    events: Sequence[Event],
    *,
    config: EngineConfig,
    deps: DependencyContext | None = None,
    presorted: bool = False,
) -> FoldResult:
    """Reduce a transaction's full event log to its reconciled state.

    ``events`` may arrive in any order; it is sorted canonically first unless
    ``presorted`` promises otherwise.
    """
    ordered = list(events) if presorted else sort_events(events)
    dependency = deps or DependencyContext(config=config)

    values: dict[str, Any] = {}
    provenance: dict[str, Provenance] = {}
    conflicts: list[dict[str, Any]] = []
    effects: dict[str, EventEffect] = {}

    exists = False
    deleted = False
    provisional = False
    delete_prov: Provenance | None = None

    for event in ordered:
        effect = _MutableEffect(event.event_id)
        prov = event.provenance

        keep_going = _apply_lifecycle(
            event=event,
            prov=prov,
            effect=effect,
            config=config,
            dependency=dependency,
            values=values,
            conflicts=conflicts,
            state=_LifecycleState(exists, deleted, provisional, delete_prov),
        )
        exists, deleted, provisional, delete_prov = (
            keep_going.exists, keep_going.deleted, keep_going.provisional, keep_going.delete_prov
        )

        if effect.blocked:
            effects[event.event_id] = effect.freeze()
            continue

        _apply_fields(
            event=event,
            prov=prov,
            effect=effect,
            config=config,
            values=values,
            provenance=provenance,
            conflicts=conflicts,
        )
        effects[event.event_id] = effect.freeze()

    # Field ownership is only final once every event has been folded, so the
    # per-event won/lost split is recomputed against the settled provenance.
    effects = _finalize_ownership(ordered, effects, provenance)

    return FoldResult(
        values=values,
        provenance=provenance,
        exists=exists,
        deleted=deleted,
        provisional=provisional,
        effects=effects,
        conflicts=tuple(conflicts),
        event_count=len(ordered),
    )


@dataclass(frozen=True, slots=True)
class _LifecycleState:
    exists: bool
    deleted: bool
    provisional: bool
    delete_prov: Provenance | None


def _apply_lifecycle(
    *,
    event: Event,
    prov: Provenance,
    effect: _MutableEffect,
    config: EngineConfig,
    dependency: DependencyContext,
    values: Mapping[str, Any],
    conflicts: list[dict[str, Any]],
    state: _LifecycleState,
) -> _LifecycleState:
    """Advance existence/deleted/provisional for one event.

    Returns the new lifecycle state; sets ``effect.blocked`` when the event must
    not touch field values at all.
    """
    exists, deleted, provisional, delete_prov = (
        state.exists, state.deleted, state.provisional, state.delete_prov
    )

    if event.action == "create":
        if deleted:
            # A create later in the timeline than the delete is a genuine
            # re-creation (a reversed void, a re-imported statement line).
            if delete_prov is None or prov.precedence > delete_prov.precedence:
                deleted, delete_prov = False, None
                effect.lifecycle.append("recreated_after_delete")
            else:
                # Outranked by the tombstone: keep its assertions as evidence
                # but leave the row deleted.
                effect.notes.append("create_outranked_by_delete")
        if provisional:
            provisional = False
            effect.lifecycle.append("provisional_confirmed")
        if not exists:
            effect.lifecycle.append("created")
        exists = True
        return _LifecycleState(exists, deleted, provisional, delete_prov)

    if event.action == "update":
        if not exists:
            # R6: the create has not arrived yet. Materialize provisionally
            # rather than discard a real financial assertion.
            if not config.allow_update_before_create:
                effect.blocked = "update_without_create"
                return _LifecycleState(exists, deleted, provisional, delete_prov)
            exists, provisional = True, True
            effect.lifecycle.append("provisional_from_update_before_create")
            return _LifecycleState(exists, deleted, provisional, delete_prov)

        if deleted:
            # R7: only a strictly higher authority may undo a delete. A routine
            # bank sync must not resurrect a row the user deliberately removed.
            higher = delete_prov is None or prov.precedence > delete_prov.precedence
            if config.resurrect_deleted_on_higher_authority and higher:
                deleted, delete_prov = False, None
                effect.lifecycle.append("restored_by_higher_authority_update")
            else:
                effect.blocked = "update_after_delete_superseded"
        return _LifecycleState(exists, deleted, provisional, delete_prov)

    # --- delete ---
    if not exists:
        # R9: delete-before-create. Record the tombstone so the late create
        # reconciles against it instead of silently resurrecting the row.
        if not config.allow_delete_before_create:
            effect.blocked = "delete_without_create"
            return _LifecycleState(exists, deleted, provisional, delete_prov)
        exists, provisional = True, True
        effect.lifecycle.append("tombstone_before_create")

    blockers = dependency.blockers(event, values.get("date"))
    if blockers:
        # R8: refuse the delete, but record it loudly. Silently dropping it
        # would leave the user believing the transaction was removed.
        effect.blocked = "delete_blocked_by_dependencies"
        effect.notes.extend(str(item["type"]) for item in blockers)
        conflicts.append({
            "type": "delete_blocked",
            "event_id": event.event_id,
            "transaction_id": event.transaction_id,
            "blockers": list(blockers),
        })
        return _LifecycleState(exists, deleted, provisional, delete_prov)

    if not deleted:
        effect.lifecycle.append("deleted")
    deleted, delete_prov = True, prov
    return _LifecycleState(exists, deleted, provisional, delete_prov)


def _apply_fields(
    *,
    event: Event,
    prov: Provenance,
    effect: _MutableEffect,
    config: EngineConfig,
    values: dict[str, Any],
    provenance: dict[str, Provenance],
    conflicts: list[dict[str, Any]],
) -> None:
    """Contest each field group this event asserted."""
    for group, members in FIELD_GROUPS:
        trigger = _GROUP_TRIGGER[group]
        if not event.asserts(trigger):
            continue

        incoming = event.value_of(trigger)
        if config.blank_never_overwrites and group in TEXT_FIELDS and not incoming:
            # Defensive: normalization already maps blanks to "not supplied", so
            # a blank category can never erase a known one.
            effect.notes.append(f"blank_ignored:{group}")
            continue

        current_prov = provenance.get(trigger)
        current = values.get(trigger)

        if group == "date":
            wins, note = _date_wins(prov, current_prov, incoming, current, config)
        else:
            wins, note = _authority_wins(prov, current_prov)

        if note:
            effect.notes.append(f"{note}:{group}")

        differs = incoming != current
        if differs and prov.ties_with(current_prov):
            # R11: same authority, same instant, different values. The winner is
            # decided by the content-derived event id - stable, but arbitrary,
            # so it is surfaced as a conflict for human review.
            conflicts.append({
                "type": "authority_tie",
                "field": group,
                "resolved_by": R11_TIE_BROKEN,
                "event_ts": format_ts(prov.event_ts),
                "authority": prov.authority,
                "candidates": sorted([
                    _render(trigger, current, values.get("currency")),
                    _render(trigger, incoming, event.currency),
                ]),
                "winner_event_id": event.event_id if wins else (
                    current_prov.event_id if current_prov else event.event_id
                ),
            })

        if group == "amount" and differs and current_prov is not None:
            existing_currency = values.get("currency")
            if existing_currency and existing_currency != event.currency:
                # Never silently convert: an FX rate is a business decision, not
                # a reconciliation one.
                conflicts.append({
                    "type": "currency_mismatch",
                    "field": "currency",
                    "existing": existing_currency,
                    "incoming": event.currency,
                    "event_id": event.event_id,
                    "note": "amounts left unconverted; review required",
                })

        if not wins:
            effect.lost.append(group)
            continue

        for member in members:
            if event.asserts(member) or member == trigger:
                values[member] = event.value_of(member)
                provenance[member] = prov
            elif member == "currency":
                # Amount and currency move as a unit even when the row relied on
                # the default currency, so minor units are never reinterpreted.
                values[member] = event.currency
                provenance[member] = prov
        effect.won.append(group)


def _authority_wins(incoming: Provenance, current: Provenance | None) -> tuple[bool, str]:
    """Default contest: strictly greater precedence takes the field."""
    if current is None:
        return True, ""
    return incoming.beats(current), ""


def _date_wins(
    incoming: Provenance,
    current: Provenance | None,
    incoming_date: Date | None,
    current_date: Date | None,
    config: EngineConfig,
) -> tuple[bool, str]:
    """Resolve a contested transaction date.

    The PRD asks for two things that pull in opposite directions: "merge with
    existing record if ``date`` is earlier" and "prefer user edits over bank
    syncs". Applying earliest-wins unconditionally would mean a user could never
    correct a too-early bank posting date, which breaks the second rule.

    The engine resolves it by scope: **earliest-wins settles ties within one
    authority level; a strictly higher authority may move the date in either
    direction.** So competing bank syncs converge on the earliest posting date
    (the classic pending-vs-posted case), while a user correction still wins
    outright. Earliest-wins is also commutative, so this keeps the fold
    order-independent.
    """
    if current is None or current_date is None:
        return True, ""
    if incoming.authority > current.authority:
        return True, "higher_authority_moved_date"
    if incoming.authority < current.authority:
        return False, "lower_authority_date_rejected"

    if not config.prefer_earliest_date_on_equal_authority:
        return incoming.beats(current), ""

    if incoming_date is None:
        return False, ""
    if incoming_date < current_date:
        return True, "earlier_date_preferred"
    if incoming_date > current_date:
        return False, "later_date_rejected_equal_authority"
    # Same date: let precedence settle provenance without changing the value.
    return incoming.beats(current), ""


def _render(field_name: str, value: Any, currency: str | None) -> str:
    """Human-readable rendering of a contested value, for conflict payloads."""
    if value is None:
        return "null"
    if field_name == "amount_minor":
        return f"{format_amount(int(value), currency or 'USD')} {currency or ''}".strip()
    if isinstance(value, Date):
        return value.isoformat()
    return str(value)


def _finalize_ownership(
    ordered: Sequence[Event],
    effects: Mapping[str, EventEffect],
    provenance: Mapping[str, Provenance],
) -> dict[str, EventEffect]:
    """Recompute won/lost against the settled provenance.

    During the fold an event can win a field and then be outranked by a later
    one. Only the final provenance says who actually owns each field, and an
    audit record that claimed otherwise would be misleading.
    """
    owners: dict[str, set[str]] = {}
    for group, _members in FIELD_GROUPS:
        trigger = _GROUP_TRIGGER[group]
        prov = provenance.get(trigger)
        if prov is not None:
            owners.setdefault(prov.event_id, set()).add(group)

    finalized: dict[str, EventEffect] = {}
    for event in ordered:
        effect = effects.get(event.event_id)
        if effect is None:  # pragma: no cover - every folded event has an effect
            continue
        if effect.blocked:
            finalized[event.event_id] = effect
            continue
        held = owners.get(event.event_id, set())
        asserted = {
            group for group, _ in FIELD_GROUPS
            if event.asserts(_GROUP_TRIGGER[group])
        }
        finalized[event.event_id] = EventEffect(
            event_id=effect.event_id,
            won=tuple(sorted(held)),
            lost=tuple(sorted(asserted - held)),
            blocked=effect.blocked,
            lifecycle=effect.lifecycle,
            notes=effect.notes,
        )
    return finalized


# ---------------------------------------------------------------------------
# Decision labelling
# ---------------------------------------------------------------------------

#: Why an event was not applied -> (decision, rule, reason).
_BLOCKED_LABELS: dict[str, tuple[str, str, str]] = {
    "delete_blocked_by_dependencies": (
        "ignored", R8_DELETE_BLOCKED, "delete_blocked_by_dependencies",
    ),
    "update_after_delete_superseded": (
        "ignored", R7_DELETE_APPLIED, "update_after_delete_superseded",
    ),
    "update_without_create": (
        "ignored", R6_UPDATE_BEFORE_CREATE, "update_without_create_rejected",
    ),
    "delete_without_create": (
        "ignored", R9_DELETE_BEFORE_CREATE, "delete_without_create_rejected",
    ),
}


@dataclass(frozen=True, slots=True)
class DecisionLabel:
    decision: str
    rule: str
    reason: str


def label_decision(
    *,
    event: Event,
    effect: EventEffect,
    state_diff: Mapping[str, Any],
    existed_before: bool,
    fold: FoldResult,
    is_late: bool,
) -> DecisionLabel:
    """Choose the ``decision``/``rule``/``reason`` triple for one event.

    ``decision`` uses the PRD's four labels plus ``created`` and ``rejected``
    (see :mod:`zoro_engine.config`). ``reason`` is a ``;``-joined list of codes,
    which is where cross-cutting annotations like ``late_event_reordered``
    appear - a late event does not replace the action's rule, it changes where in
    the timeline that rule was applied.
    """
    reasons: list[str] = []
    if is_late:
        reasons.append("late_event_reordered")
    reasons.extend(effect.notes)

    if effect.blocked:
        decision, rule, primary = _BLOCKED_LABELS.get(
            effect.blocked, ("ignored", R5_UPDATE_MERGE, effect.blocked)
        )
        return DecisionLabel(decision, rule, _join([primary, *reasons]))

    op = state_diff.get("op", "none")
    rule = _rule_for(event, effect, existed_before)

    if op == "created":
        return DecisionLabel("created", rule, _join(["transaction_created", *reasons]))

    if op == "deleted":
        return DecisionLabel("deleted", R7_DELETE_APPLIED, _join(["transaction_deleted", *reasons]))

    if op == "restored":
        return DecisionLabel("merged", rule, _join(["transaction_restored", *reasons]))

    if diff_is_empty(state_diff):
        # Nothing about the money changed. Distinguishing *why* matters: a
        # superseded late event and a re-sent identical row are very different
        # events to see in an audit trail.
        if is_late:
            primary = "late_event_superseded"
        elif effect.lost and not effect.won:
            primary = "outranked_by_existing_values"
        else:
            primary = "no_material_change"
        return DecisionLabel("ignored", rule, _join([primary, *reasons]))

    # Something changed. "replaced" means this event now owns every materialized
    # field group; anything less is a merge of competing sources.
    owned_all = _owns_everything(effect, fold)
    if owned_all:
        return DecisionLabel("replaced", rule, _join(["all_fields_from_this_event", *reasons]))

    detail = f"fields_merged:{','.join(effect.won)}" if effect.won else "fields_unchanged"
    if effect.lost:
        detail += f";outranked_on:{','.join(effect.lost)}"
    return DecisionLabel("merged", rule, _join([detail, *reasons]))


def _rule_for(event: Event, effect: EventEffect, existed_before: bool) -> str:
    """The primary action rule that governed this event."""
    if event.action == "create":
        return R3_CREATE_NEW if not existed_before else R4_CREATE_DUPLICATE
    if event.action == "update":
        if "provisional_from_update_before_create" in effect.lifecycle:
            return R6_UPDATE_BEFORE_CREATE
        return R5_UPDATE_MERGE
    if "tombstone_before_create" in effect.lifecycle:
        return R9_DELETE_BEFORE_CREATE
    return R7_DELETE_APPLIED


def _owns_everything(effect: EventEffect, fold: FoldResult) -> bool:
    """True if this event's values hold every field group that has a value."""
    materialized = {
        group for group, _ in FIELD_GROUPS
        if fold.values.get(_GROUP_TRIGGER[group]) is not None
    }
    if not materialized:
        return False
    return materialized.issubset(set(effect.won))


def _join(parts: Iterable[str]) -> str:
    """Join reason codes, dropping blanks and preserving first-seen order."""
    seen: dict[str, None] = {}
    for part in parts:
        if part:
            seen.setdefault(part, None)
    return ";".join(seen)


def build_evidence(
    events: Sequence[Event],
    *,
    fold: FoldResult,
    limit: int = MAX_EVIDENCE_EVENTS,
) -> tuple[dict[str, Any], ...]:
    """The ``evidence`` list: every event considered, with its outcome.

    Ordered by the canonical timeline so a reviewer reads the transaction's
    history in the order the engine reasoned about it.
    """
    ordered = sort_events(events)
    truncated = len(ordered) > limit
    window = ordered[-limit:] if truncated else ordered

    items: list[dict[str, Any]] = []
    if truncated:
        items.append({
            "note": "evidence_truncated",
            "total_events": len(ordered),
            "shown": len(window),
        })

    for event in window:
        item = event.to_evidence()
        effect = fold.effect_for(event.event_id)
        if effect.won:
            item["won"] = list(effect.won)
        if effect.lost:
            item["lost"] = list(effect.lost)
        if effect.blocked:
            item["blocked"] = effect.blocked
        if effect.lifecycle:
            item["lifecycle"] = list(effect.lifecycle)
        items.append(item)
    return tuple(items)
