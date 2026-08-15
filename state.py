"""Versioned per-user state, dependency indexing and financial rollups.

Holds three things per user:

* **the accepted event log** - in arrival order, which is what makes
  version-by-version reconstruction possible;
* **the materialized transactions** - each one the output of folding its own
  event log (see :mod:`zoro_engine.resolution`);
* **a live-children index** - so R8 can answer "does anything depend on this
  transaction?" without scanning every record on every delete.

The rollups in :meth:`UserState.derived` are the reason the engine exists: the
PRD's requirement that "all financial models reflect only the most accurate and
up-to-date state" means something concrete only if the reconciled state is
actually aggregated somewhere. Deleted transactions drop out of every total;
provisional ones (R6) are included but counted separately so a downstream model
can exclude them on its own terms.
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass, replace
from datetime import datetime
from typing import Any, Iterable, Iterator, Mapping, Sequence

from .canonical import canonical_json, sha256_hex
from .config import DEFAULT_CONFIG, EngineConfig
from .models import (
    Event,
    Transaction,
    diff_is_empty,
    diff_snapshots,
)
from .money import format_amount
from .normalize import identity_key
from .resolution import (
    DependencyContext,
    EventEffect,
    FoldResult,
    fold_transaction,
)

#: Bucket label for transactions with no usable date.
UNDATED = "undated"
#: Bucket label for transactions with no category.
UNCATEGORIZED = "uncategorized"


# ---------------------------------------------------------------------------
# Apply outcome
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class ApplyOutcome:
    """Everything the audit layer needs to describe one applied event."""

    event: Event
    transaction: Transaction | None
    before: dict[str, Any] | None
    after: dict[str, Any] | None
    state_diff: dict[str, Any]
    fold: FoldResult
    effect: EventEffect
    is_late: bool
    considered: tuple[Event, ...]
    new_conflicts: tuple[dict[str, Any], ...]
    user_version: int
    txn_version: int
    state_hash: str

    @property
    def changed(self) -> bool:
        return not diff_is_empty(self.state_diff)

    @property
    def existed_before(self) -> bool:
        return self.before is not None


# ---------------------------------------------------------------------------
# Per-user state
# ---------------------------------------------------------------------------

class UserState:
    """One user's reconciled financial history.

    Not a dataclass: it owns mutable indices whose invariants are maintained by
    :meth:`apply`, and exposing them as plain fields would invite the kind of
    out-of-band mutation that silently desynchronizes the children index.
    """

    __slots__ = (
        "_children", "_txn_conflicts", "accepted", "config",
        "logs", "transactions", "user_id", "user_key", "version",
    )

    def __init__(
        self,
        user_id: str,
        user_key: str | None = None,
        config: EngineConfig = DEFAULT_CONFIG,
    ) -> None:
        self.user_id = user_id
        self.user_key = user_key if user_key is not None else identity_key(
            user_id, case_insensitive=config.case_insensitive_ids
        )
        self.config = config
        self.version = 0
        #: transaction_key -> materialized transaction
        self.transactions: dict[str, Transaction] = {}
        #: transaction_key -> that transaction's event log, canonically sorted
        self.logs: dict[str, list[Event]] = {}
        #: every accepted event, in arrival order (drives version replay)
        self.accepted: list[Event] = []
        #: parent_key -> live child transaction keys
        self._children: dict[str, set[str]] = {}
        #: transaction_key -> conflicts from that transaction's latest fold
        self._txn_conflicts: dict[str, tuple[dict[str, Any], ...]] = {}

    # --- dependency lookup (R8) ------------------------------------------

    def live_children(self, _user_key: str, txn_key: str) -> Sequence[str]:
        """Display ids of live transactions naming ``txn_key`` as their parent."""
        keys = self._children.get(txn_key)
        if not keys:
            return ()
        return tuple(sorted(
            self.transactions[key].transaction_id
            for key in keys
            if key in self.transactions
        ))

    def _parent_key_of(self, txn: Transaction | None) -> str | None:
        """The parent key a transaction currently depends on, if it is live.

        A deleted transaction depends on nothing, so deleting a child releases
        the lock on its parent automatically.
        """
        if txn is None or txn.deleted or not txn.parent_transaction_id:
            return None
        return identity_key(
            txn.parent_transaction_id, case_insensitive=self.config.case_insensitive_ids
        )

    def _reindex(self, old: Transaction | None, new: Transaction | None) -> None:
        """Keep the children index in step with a transaction's new state."""
        key = (new or old).transaction_key if (new or old) else None
        if key is None:  # pragma: no cover - defensive
            return
        old_parent = self._parent_key_of(old)
        new_parent = self._parent_key_of(new)
        if old_parent == new_parent:
            return
        if old_parent is not None:
            siblings = self._children.get(old_parent)
            if siblings is not None:
                siblings.discard(key)
                if not siblings:
                    del self._children[old_parent]
        if new_parent is not None:
            self._children.setdefault(new_parent, set()).add(key)

    # --- applying events --------------------------------------------------

    def apply(self, event: Event) -> ApplyOutcome:
        """Fold ``event`` into this user's state and report what changed.

        The transaction's entire log is re-folded rather than patched. That is
        deliberate: it is what makes a late event correct instead of merely
        detected, and it means the materialized state can never drift from the
        events that produced it.
        """
        if event.user_key != self.user_key:
            raise ValueError(
                f"event for user {event.user_key!r} applied to state for {self.user_key!r}"
            )

        txn_key = event.transaction_key
        log = self.logs.setdefault(txn_key, [])

        # A late event is one that sorts before something already folded. The
        # fold handles it correctly either way; we detect it only so the audit
        # record can say so.
        is_late = bool(log) and event.canonical_key < log[-1].canonical_key
        bisect.insort(log, event, key=lambda item: item.canonical_key)

        before_txn = self.transactions.get(txn_key)
        before = before_txn.snapshot() if before_txn is not None else None

        fold = fold_transaction(
            log,
            config=self.config,
            deps=DependencyContext(config=self.config, children=self.live_children),
            presorted=True,
        )

        # Build at the old version first: snapshot() excludes version, so the
        # diff is unaffected, and the version only advances if something moved.
        old_version = before_txn.version if before_txn is not None else 0
        tentative = fold.build(
            user_id=self.user_id,
            user_key=self.user_key,
            transaction_id=before_txn.transaction_id if before_txn else event.transaction_id,
            transaction_key=txn_key,
            version=old_version,
            default_currency=self.config.default_currency,
        )
        after = tentative.snapshot() if tentative is not None else None
        state_diff = diff_snapshots(before, after)
        changed = not diff_is_empty(state_diff)

        final = (
            replace(tentative, version=old_version + 1)
            if tentative is not None and changed
            else tentative
        )

        if final is not None:
            self.transactions[txn_key] = final
        if changed:
            self.version += 1

        self._reindex(before_txn, final)
        self.accepted.append(event)

        new_conflicts = self._diff_conflicts(txn_key, fold.conflicts)
        self._txn_conflicts[txn_key] = fold.conflicts

        return ApplyOutcome(
            event=event,
            transaction=final,
            before=before,
            after=after,
            state_diff=state_diff,
            fold=fold,
            effect=fold.effect_for(event.event_id),
            is_late=is_late,
            considered=tuple(log),
            new_conflicts=new_conflicts,
            user_version=self.version,
            txn_version=final.version if final is not None else old_version,
            state_hash=self.state_hash(),
        )

    def _diff_conflicts(
        self, txn_key: str, conflicts: tuple[dict[str, Any], ...]
    ) -> tuple[dict[str, Any], ...]:
        """Conflicts this fold surfaced that the previous fold had not.

        The fold recomputes a transaction's *entire* conflict set each time, so
        reporting all of them on every event would make one long-running tie
        look like a hundred separate problems.
        """
        previous = {
            canonical_json(item) for item in self._txn_conflicts.get(txn_key, ())
        }
        return tuple(item for item in conflicts if canonical_json(item) not in previous)

    # --- views ------------------------------------------------------------

    def live_transactions(self) -> list[Transaction]:
        """Transactions visible to financial models, in stable key order."""
        return [
            txn for _key, txn in sorted(self.transactions.items())
            if not txn.deleted
        ]

    def financial_view(self) -> dict[str, Any]:
        """The hashed view: reconciled money only, no bookkeeping."""
        return {
            txn.transaction_id: txn.snapshot()
            for _key, txn in sorted(self.transactions.items())
        }

    def authority_view(self) -> dict[str, Any]:
        """Financial state *plus* per-field provenance."""
        return {
            txn.transaction_id: txn.to_full()
            for _key, txn in sorted(self.transactions.items())
        }

    def state_hash(self) -> str:
        """Hash of the reconciled financial state after the last event."""
        return sha256_hex(self.financial_view())

    def authority_hash(self) -> str:
        """Hash of state *and* provenance.

        Verified separately on replay: two runs can agree on every amount while
        disagreeing about *who* set it, and that divergence would change how
        future conflicts resolve.
        """
        return sha256_hex(self.authority_view())

    def conflicts(self) -> list[dict[str, Any]]:
        """Every unresolved conflict currently flagged for this user."""
        items: list[dict[str, Any]] = []
        for txn_key in sorted(self._txn_conflicts):
            for conflict in self._txn_conflicts[txn_key]:
                items.append({"transaction_key": txn_key, **conflict})
        return items

    def snapshot(self) -> dict[str, Any]:
        """Full user state, as written to ``final_state.json``."""
        return {
            "user_id": self.user_id,
            "role": self.config.role_for(self.user_id),
            "version": self.version,
            "event_count": len(self.accepted),
            "state_hash": self.state_hash(),
            "authority_hash": self.authority_hash(),
            "transactions": {
                txn.transaction_id: txn.to_public()
                for _key, txn in sorted(self.transactions.items())
            },
            "derived": self.derived(),
            "conflicts": self.conflicts(),
        }

    # --- financial rollups ------------------------------------------------

    def derived(self) -> dict[str, Any]:
        """Aggregates recomputed from the reconciled state.

        Sign convention: **positive is inflow, negative is outflow**. The engine
        does not impose a spend convention on the feed - it reports what the
        amounts say, grouped by currency so that unlike units are never summed.

        Deleted transactions are excluded entirely. Provisional ones (R6) are
        included but counted separately, because dropping a real financial
        assertion just because its create is still in flight would understate a
        user's spending.
        """
        counts = {"live": 0, "deleted": 0, "provisional": 0, "unpriced": 0}
        by_currency: dict[str, dict[str, int]] = {}
        by_month: dict[str, dict[str, dict[str, int]]] = {}
        by_category: dict[str, dict[str, dict[str, int]]] = {}

        for _key, txn in sorted(self.transactions.items()):
            if txn.deleted:
                counts["deleted"] += 1
                continue
            counts["live"] += 1
            if txn.provisional:
                counts["provisional"] += 1
            if txn.amount_minor is None:
                counts["unpriced"] += 1
                continue

            amount = txn.amount_minor
            currency = txn.currency

            totals = by_currency.setdefault(
                currency,
                {"net_minor": 0, "inflow_minor": 0, "outflow_minor": 0, "count": 0},
            )
            totals["net_minor"] += amount
            totals["count"] += 1
            if amount >= 0:
                totals["inflow_minor"] += amount
            else:
                totals["outflow_minor"] += amount

            month = txn.date.strftime("%Y-%m") if txn.date else UNDATED
            bucket = by_month.setdefault(month, {}).setdefault(
                currency, {"net_minor": 0, "count": 0}
            )
            bucket["net_minor"] += amount
            bucket["count"] += 1

            category = txn.category or UNCATEGORIZED
            bucket = by_category.setdefault(category, {}).setdefault(
                currency, {"net_minor": 0, "count": 0}
            )
            bucket["net_minor"] += amount
            bucket["count"] += 1

        return {
            "counts": counts,
            "by_currency": {
                currency: _with_formatted(values, currency)
                for currency, values in sorted(by_currency.items())
            },
            "by_month": _format_nested(by_month),
            "by_category": _format_nested(by_category),
        }

    # --- reconstruction over time ----------------------------------------

    def rebuild_from(self, events: Iterable[Event]) -> UserState:
        """A fresh state built by applying ``events`` in the order given."""
        fresh = UserState(self.user_id, self.user_key, self.config)
        for event in events:
            fresh.apply(event)
        return fresh

    def at_version(self, version: int) -> UserState:
        """State as the system believed it after reaching ``version``.

        *Processing-time* reconstruction: replays the accepted log in arrival
        order and stops once the version counter reaches the target. Answers
        "what did the platform think at the time?", which is the question an
        auditor asks about a decision that has since been superseded.
        """
        if version < 0:
            raise ValueError(f"version must be non-negative, got {version}")
        fresh = UserState(self.user_id, self.user_key, self.config)
        if version == 0:
            return fresh
        for event in self.accepted:
            fresh.apply(event)
            if fresh.version >= version:
                break
        return fresh

    def as_of(self, moment: datetime) -> UserState:
        """State considering only events *asserted* at or before ``moment``.

        *Business-time* reconstruction: answers "what was actually true as of
        this date?" - which differs from :meth:`at_version` precisely when
        events arrived late, and is the view a restated financial report needs.
        """
        return self.rebuild_from(
            event for event in self.accepted if event.event_ts <= moment
        )

    def __repr__(self) -> str:  # pragma: no cover - display only
        return (
            f"UserState(user_id={self.user_id!r}, version={self.version}, "
            f"transactions={len(self.transactions)}, events={len(self.accepted)})"
        )


def _with_formatted(values: Mapping[str, int], currency: str) -> dict[str, Any]:
    """Add exact string renderings alongside minor-unit integers.

    Both forms are emitted on purpose: the minor units are what downstream code
    should compute with, the strings are what a human (or Power BI) should read.
    """
    payload: dict[str, Any] = dict(sorted(values.items()))
    for name in ("net_minor", "inflow_minor", "outflow_minor"):
        if name in payload:
            payload[name.removesuffix("_minor")] = format_amount(payload[name], currency)
    payload["currency"] = currency
    return payload


def _format_nested(
    buckets: Mapping[str, Mapping[str, Mapping[str, int]]],
) -> dict[str, Any]:
    return {
        label: {
            currency: _with_formatted(values, currency)
            for currency, values in sorted(per_currency.items())
        }
        for label, per_currency in sorted(buckets.items())
    }


# ---------------------------------------------------------------------------
# All users
# ---------------------------------------------------------------------------

class StateBook:
    """Every user's state, keyed by identity form of ``user_id``."""

    __slots__ = ("config", "users")

    def __init__(self, config: EngineConfig = DEFAULT_CONFIG) -> None:
        self.config = config
        self.users: dict[str, UserState] = {}

    def for_event(self, event: Event) -> UserState:
        """The state for ``event``'s user, created on first sight."""
        state = self.users.get(event.user_key)
        if state is None:
            state = UserState(event.user_id, event.user_key, self.config)
            self.users[event.user_key] = state
        return state

    def get(self, user_id: str) -> UserState | None:
        """Look up by display id or identity key."""
        key = identity_key(user_id, case_insensitive=self.config.case_insensitive_ids)
        return self.users.get(key) or self.users.get(user_id)

    def apply(self, event: Event) -> ApplyOutcome:
        return self.for_event(event).apply(event)

    def __iter__(self) -> Iterator[UserState]:
        for key in sorted(self.users):
            yield self.users[key]

    def __len__(self) -> int:
        return len(self.users)

    def snapshot(self) -> dict[str, Any]:
        """Cross-user final state, as written to ``final_state.json``."""
        users = {state.user_id: state.snapshot() for state in self}
        return {
            "users": users,
            "totals": {
                "users": len(users),
                "transactions": sum(len(state.transactions) for state in self),
                "live_transactions": sum(len(state.live_transactions()) for state in self),
                "events": sum(len(state.accepted) for state in self),
                "conflicts": sum(len(state.conflicts()) for state in self),
            },
            "state_hash": self.state_hash(),
        }

    def state_hash(self) -> str:
        """Hash across every user - the platform-wide determinism check."""
        return sha256_hex({state.user_id: state.state_hash() for state in self})
