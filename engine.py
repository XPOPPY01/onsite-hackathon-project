"""The engine: the orchestrator that turns raw rows into audited decisions.

One event's journey:

1. **normalize** - raw row to :class:`~zoro_engine.models.Event`, or a rejection
   carrying every problem found (R2, HTTP 400).
2. **duplicate check** - the content-addressed ``event_id`` against the store's
   ledger (R1, HTTP 409).
3. **apply** - the event joins its transaction's log, which is then re-folded
   from scratch (:mod:`zoro_engine.resolution`).
4. **label** - the fold's effect becomes a ``decision``/``rule``/``reason``.
5. **detect** - advisory anomaly flags (bonus), which never alter the outcome.
6. **record** - appended to the hash-chained audit log and the store.

Steps 3-6 are pure functions of the accepted event set and the config. Nothing
consults the wall clock, a random source, or dictionary iteration order, which is
what makes :meth:`Engine.verify_replay` able to assert byte-equality rather than
mere similarity.

The engine deliberately does **not** own I/O beyond the store and audit sink:
readers live in :mod:`zoro_engine.ingest`, the HTTP surface in
:mod:`zoro_engine.api`, and the command line in :mod:`zoro_engine.cli`. Each of
those is a thin adapter over this class.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping, Sequence

from . import __version__
from .anomaly import detect_anomalies
from .audit import AuditSink, MemoryAuditSink, build_header, head_hash, summarize
from .canonical import format_ts, stable_id
from .config import DEFAULT_CONFIG, EngineConfig
from .errors import ValidationError
from .ingest import RawRow
from .models import Decision, Event, Issue, build_decision
from .normalize import normalize_row
from .resolution import build_evidence, label_decision
from .state import StateBook, UserState
from .store import MemoryStore, Store

#: HTTP-ish status codes, as the PRD specifies for ``POST /events``.
STATUS_OK = 200
STATUS_BAD_REQUEST = 400
STATUS_DUPLICATE = 409

#: Rule identifiers owned by the engine rather than the fold.
R1_DUPLICATE_EVENT = "R1_DUPLICATE_EVENT"
R2_SCHEMA_REJECT = "R2_SCHEMA_REJECT"


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class EventResult:
    """The outcome of offering one row to the engine."""

    status: int
    decision: Decision | None = None
    issues: tuple[dict[str, Any], ...] = ()
    origin: str = ""
    event_id: str = ""

    @property
    def accepted(self) -> bool:
        return self.status == STATUS_OK

    @property
    def duplicate(self) -> bool:
        return self.status == STATUS_DUPLICATE

    @property
    def rejected(self) -> bool:
        return self.status == STATUS_BAD_REQUEST

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"status": self.status, "event_id": self.event_id}
        if self.origin:
            payload["origin"] = self.origin
        if self.decision is not None:
            payload["decision"] = self.decision.to_response()
        if self.issues:
            payload["issues"] = list(self.issues)
        return payload


@dataclass(frozen=True, slots=True)
class BatchResult:
    """The outcome of a whole upload."""

    results: tuple[EventResult, ...] = ()

    @property
    def accepted(self) -> tuple[EventResult, ...]:
        return tuple(item for item in self.results if item.accepted)

    @property
    def duplicates(self) -> tuple[EventResult, ...]:
        return tuple(item for item in self.results if item.duplicate)

    @property
    def rejected(self) -> tuple[EventResult, ...]:
        return tuple(item for item in self.results if item.rejected)

    @property
    def decisions(self) -> tuple[Decision, ...]:
        return tuple(item.decision for item in self.results if item.decision is not None)

    @property
    def status(self) -> int:
        """A single status for the batch.

        A partial success is still a success: rejecting an entire 10,000-row
        upload because three lines were malformed would be the wrong contract for
        a reconciliation tool, and the per-row statuses are always returned
        alongside. ``400`` and ``409`` are reserved for batches where *nothing*
        was accepted, which is the single-event case the PRD describes.
        """
        if not self.results:
            return STATUS_BAD_REQUEST
        if self.accepted:
            return STATUS_OK
        if self.rejected:
            return STATUS_BAD_REQUEST
        return STATUS_DUPLICATE

    def counts(self) -> dict[str, int]:
        return {
            "received": len(self.results),
            "accepted": len(self.accepted),
            "duplicate": len(self.duplicates),
            "rejected": len(self.rejected),
        }

    def to_dict(self, *, include_events: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {"status": self.status, "counts": self.counts()}
        if include_events:
            payload["results"] = [item.to_dict() for item in self.results]
        else:
            # Failures are always listed: a summary that hides them would let a
            # broken feed look like a clean import.
            payload["failures"] = [
                item.to_dict() for item in self.results if not item.accepted
            ]
        return payload


@dataclass(frozen=True, slots=True)
class Mismatch:
    """One field where a replayed decision diverged from the recorded one."""

    seq: int
    event_id: str
    field: str
    recorded: Any
    replayed: Any

    def to_dict(self) -> dict[str, Any]:
        return {
            "seq": self.seq,
            "event_id": self.event_id,
            "field": self.field,
            "recorded": self.recorded,
            "replayed": self.replayed,
        }


@dataclass(frozen=True, slots=True)
class VerifyReport:
    """Result of replaying a run and comparing it to what was recorded."""

    ok: bool
    events: int
    compared: int
    mismatches: tuple[Mismatch, ...] = ()
    state_hash_recorded: str = ""
    state_hash_replayed: str = ""
    audit_head_recorded: str = ""
    audit_head_replayed: str = ""
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "ok": self.ok,
            "events_replayed": self.events,
            "decisions_compared": self.compared,
            "state_hash": {
                "recorded": self.state_hash_recorded,
                "replayed": self.state_hash_replayed,
                "match": self.state_hash_recorded == self.state_hash_replayed,
            },
            "audit_head": {
                "recorded": self.audit_head_recorded,
                "replayed": self.audit_head_replayed,
                "match": self.audit_head_recorded == self.audit_head_replayed,
            },
        }
        if self.mismatches:
            payload["mismatches"] = [item.to_dict() for item in self.mismatches]
        if self.detail:
            payload["detail"] = self.detail
        return payload

    def __str__(self) -> str:  # pragma: no cover - display only
        if self.ok:
            return (
                f"replay reproduced {self.compared} decision(s) exactly "
                f"across {self.events} event(s)"
            )
        return (
            f"replay diverged: {len(self.mismatches)} mismatch(es) "
            f"across {self.events} event(s)"
        )


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class Engine:
    """Stateful facade over normalization, resolution, state and audit."""

    __slots__ = ("_audit", "_seen", "_seq", "book", "config", "store")

    def __init__(
        self,
        *,
        config: EngineConfig = DEFAULT_CONFIG,
        store: Store | None = None,
        audit: AuditSink | None = None,
    ) -> None:
        self.config = config
        self.store = store if store is not None else MemoryStore()
        self._audit: AuditSink = audit if audit is not None else MemoryAuditSink()
        self.book = StateBook(config)

        # The chain is seeded with the config fingerprint, so a log written under
        # different rules can never be mistaken for one written under these.
        self._audit.start(build_header(config, engine_version=__version__))

        self._seq = self.store.next_seq()
        #: Mirror of the store's ledger. The store remains authoritative; this
        #: only avoids a query per incoming row on large uploads.
        self._seen: set[str] = set(self.store.seen_ids())

        # A store handed to us with existing events must be folded back in, or
        # the first new event would reconcile against an empty state and every
        # decision after it would be wrong.
        if self.store.event_count():
            for event in self.store.events():
                self.book.apply(event)

    # --- properties --------------------------------------------------------

    @property
    def audit(self) -> AuditSink:
        return self._audit

    @property
    def event_count(self) -> int:
        return self.store.event_count()

    def audit_records(self) -> Sequence[dict[str, Any]]:
        return self._audit.records()

    def audit_head(self) -> str:
        return self._audit.last_hash

    # --- ingestion ---------------------------------------------------------

    def ingest_row(
        self,
        row: Mapping[str, Any],
        *,
        origin: str = "api",
    ) -> EventResult:
        """Normalize, resolve, record. The single-event path.

        Never raises for bad *data* - a malformed row comes back as a ``400``
        result with every problem listed. Exceptions are reserved for broken
        *infrastructure* (an unwritable log, a failing store), which a caller must
        handle very differently from a bad spreadsheet cell.
        """
        result = normalize_row(row, config=self.config, origin=origin)

        if result.event is None:
            return self._reject(result.raw, result.errors, origin)

        event = result.event
        if event.event_id in self._seen:
            return self._duplicate(event, origin)

        return self._accept(event, origin)

    def ingest_rows(
        self,
        rows: Iterable[Mapping[str, Any] | RawRow],
        *,
        origin_prefix: str = "api",
    ) -> BatchResult:
        """Ingest many rows in the order given.

        Order is preserved because it *is* the arrival order the audit trail
        narrates. The fold re-sorts into timeline order internally, so arrival
        order changes the log's story without changing the final state.
        """
        results: list[EventResult] = []
        for index, item in enumerate(rows, start=1):
            if isinstance(item, RawRow):
                data, origin = item.data, item.origin
            else:
                data, origin = item, f"{origin_prefix}:{index}"
            results.append(self.ingest_row(data, origin=origin))

        self.store.commit()
        return BatchResult(results=tuple(results))

    # Convenience aliases matching the PRD's vocabulary.
    ingest_event = ingest_row
    ingest_batch = ingest_rows

    def _accept(self, event: Event, origin: str) -> EventResult:
        """Apply an accepted event and record the decision."""
        stamped = event.with_seq(self._seq)
        outcome = self.book.apply(stamped)

        label = label_decision(
            event=stamped,
            effect=outcome.effect,
            state_diff=outcome.state_diff,
            existed_before=outcome.existed_before,
            fold=outcome.fold,
            is_late=outcome.is_late,
        )

        anomalies = detect_anomalies(
            event=stamped,
            outcome=outcome,
            state=self.book.for_event(stamped),
            config=self.config,
        )

        decision = build_decision(
            seq=stamped.seq,
            event_id=stamped.event_id,
            user_id=stamped.user_id,
            # The materialized display id, which may differ in casing from the
            # incoming row: first-seen casing wins so the audit trail is stable.
            transaction_id=(
                outcome.transaction.transaction_id
                if outcome.transaction is not None
                else stamped.transaction_id
            ),
            action=stamped.action,
            source=stamped.source,
            timestamp=stamped.event_ts,
            decision=label.decision,
            rule=label.rule,
            reason=label.reason,
            state_diff=outcome.state_diff,
            evidence=build_evidence(outcome.considered, fold=outcome.fold),
            conflicts=outcome.new_conflicts,
            anomalies=anomalies,
            warnings=stamped.warnings,
            user_version=outcome.user_version,
            txn_version=outcome.txn_version,
            state_hash=outcome.state_hash,
        )

        # Store before audit: the event log is the source of truth, and an event
        # narrated in the audit trail but absent from the log would make replay
        # impossible. The reverse (stored but unnarrated) is recoverable.
        self.store.add_event(stamped)
        sealed = self._audit.append(decision)
        self.store.add_decision(sealed)

        self._seen.add(stamped.event_id)
        self._seq += 1

        return EventResult(
            status=STATUS_OK, decision=sealed, origin=origin, event_id=stamped.event_id
        )

    def _duplicate(self, event: Event, origin: str) -> EventResult:
        """R1: an already-processed ``event_id``.

        Recorded in the audit trail but **not** re-applied and **not** re-stored.
        Logging it matters: "we saw this again and declined" is information a
        reviewer needs, and silently dropping it would make a misbehaving feed
        invisible.
        """
        state = self.book.get(event.user_id)
        decision = build_decision(
            seq=self._seq,
            event_id=event.event_id,
            user_id=event.user_id,
            transaction_id=event.transaction_id,
            action=event.action,
            source=event.source,
            timestamp=event.event_ts,
            decision="ignored",
            rule=R1_DUPLICATE_EVENT,
            reason="duplicate_event_id",
            state_diff={"op": "none", "fields": {}},
            user_version=state.version if state is not None else 0,
            state_hash=state.state_hash() if state is not None else "",
        )
        sealed = self._audit.append(decision)
        # Deliberately not stored: `events` is keyed by event_id and a duplicate
        # would violate that UNIQUE constraint, which is the point of it.
        self._seq += 1
        return EventResult(
            status=STATUS_DUPLICATE,
            decision=sealed,
            origin=origin,
            event_id=event.event_id,
        )

    def _reject(
        self,
        raw: Mapping[str, Any],
        errors: Sequence[Issue],
        origin: str,
    ) -> EventResult:
        """R2: the row failed normalization.

        Recorded so a malformed upload leaves a trace. The id is content-derived
        from the raw row, so the same bad row is recognizable across runs even
        though it never became an event.
        """
        issues = tuple(issue.to_dict() for issue in errors)
        reject_id = "bad-" + stable_id(
            {str(k): (None if v is None else str(v)) for k, v in raw.items()},
            length=24,
        )
        decision = build_decision(
            seq=self._seq,
            event_id=reject_id,
            # Best effort only: these fields are exactly what failed to parse, so
            # they are recorded as strings purely to help locate the bad row.
            user_id=str(raw.get("user_id") or ""),
            transaction_id=str(raw.get("transaction_id") or ""),
            action=str(raw.get("action") or ""),
            source=str(raw.get("source") or ""),
            # No usable event timestamp exists, so the epoch is used as an
            # explicit sentinel rather than the wall clock, which would make the
            # audit log non-reproducible.
            timestamp=datetime(1970, 1, 1, tzinfo=timezone.utc),
            decision="rejected",
            rule=R2_SCHEMA_REJECT,
            reason="schema_validation_failed",
            state_diff={"op": "none", "fields": {}},
            issues=issues,
        )
        sealed = self._audit.append(decision)
        self._seq += 1
        return EventResult(
            status=STATUS_BAD_REQUEST,
            issues=issues,
            origin=origin,
            event_id=reject_id,
            decision=sealed,
        )

    # --- state views -------------------------------------------------------

    def final_state(self) -> dict[str, Any]:
        """The reconciled state of every user, as written to ``final_state.json``."""
        return self.book.snapshot()

    def state_hash(self) -> str:
        return self.book.state_hash()

    def user_state(self, user_id: str) -> UserState | None:
        return self.book.get(user_id)

    def state_at(self, user_id: str, version: int) -> dict[str, Any]:
        """Processing-time reconstruction: what the platform believed at ``version``."""
        state = self.book.get(user_id)
        if state is None:
            raise ValidationError(f"unknown user {user_id!r}", field="user_id")
        return state.at_version(version).snapshot()

    def state_as_of(self, user_id: str, moment: datetime) -> dict[str, Any]:
        """Business-time reconstruction: what was true as of ``moment``."""
        state = self.book.get(user_id)
        if state is None:
            raise ValidationError(f"unknown user {user_id!r}", field="user_id")
        return state.as_of(moment).snapshot()

    def conflicts(self) -> list[dict[str, Any]]:
        """Every unresolved conflict across all users."""
        items: list[dict[str, Any]] = []
        for state in self.book:
            for conflict in state.conflicts():
                items.append({"user_id": state.user_id, **conflict})
        return items

    def anomalies(self) -> list[dict[str, Any]]:
        """Every anomaly flag recorded in this run's audit trail."""
        items: list[dict[str, Any]] = []
        for record in self._audit.records():
            for anomaly in record.get("anomalies") or ():
                items.append({
                    "event_id": record.get("event_id"),
                    "user_id": record.get("user_id"),
                    "transaction_id": record.get("transaction_id"),
                    **anomaly,
                })
        return items

    def summary(self) -> dict[str, Any]:
        """Counts by decision, rule and reason, plus state totals."""
        payload = summarize(self._audit.records())
        payload["state_hash"] = self.state_hash()
        payload["audit_head"] = self.audit_head()
        payload["config_fingerprint"] = self.config.fingerprint()
        return payload

    # --- persistence -------------------------------------------------------

    def save_state(self) -> None:
        """Write the materialized state to the store's reporting table."""
        rows = [
            txn
            for state in self.book
            for _key, txn in sorted(state.transactions.items())
        ]
        self.store.save_transactions(rows)
        self.store.commit()

    def run_manifest(
        self,
        *,
        run_id: str,
        started_at: datetime,
        finished_at: datetime,
        counts: Mapping[str, int] | None = None,
        notes: str = "",
    ) -> dict[str, Any]:
        """Wall-clock run metadata.

        Kept strictly out of ``audit.log``: the moment a real timestamp enters the
        hashed trail, two identical runs stop producing identical bytes and
        ``verify`` can no longer assert byte-equality. Timing belongs here.
        """
        tally = dict(counts or {})
        return {
            "run_id": run_id,
            "started_at": format_ts(started_at),
            "finished_at": format_ts(finished_at),
            "duration_seconds": round(
                (finished_at - started_at).total_seconds(), 6
            ),
            "engine_version": __version__,
            "config_fingerprint": self.config.fingerprint(),
            "events_accepted": tally.get("accepted", 0),
            "events_rejected": tally.get("rejected", 0),
            "events_duplicate": tally.get("duplicate", 0),
            "audit_head_hash": self.audit_head(),
            "state_hash": self.state_hash(),
            "notes": notes,
        }

    def close(self) -> None:
        self._audit.close()
        self.store.close()

    def __enter__(self) -> Engine:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def __repr__(self) -> str:  # pragma: no cover - display only
        return (
            f"Engine(events={self.event_count}, users={len(self.book)}, "
            f"state_hash={self.state_hash()[:12]}...)"
        )

    # --- replay ------------------------------------------------------------

    def replay(self, *, config: EngineConfig | None = None) -> Engine:
        """Rebuild a fresh engine from the stored event log.

        The events are fed in as **already-normalized objects**, skipping the
        normalization step - deliberately. Replay's job is to prove the
        *resolution* rules are deterministic; re-parsing the original strings
        would conflate a normalization change with a rule change, and the raw
        text is preserved in ``events.raw`` for that separate check.

        Writes to a throwaway in-memory store and audit sink. Replaying into the
        live ``audit.log`` would corrupt the very history being verified.
        """
        replayed = Engine(
            config=config or self.config,
            store=MemoryStore(),
            audit=MemoryAuditSink(),
        )
        for event in self.store.events():
            replayed._replay_one(event)
        return replayed

    def _replay_one(self, event: Event) -> Decision:
        """Re-apply a stored event, preserving its original sequence number.

        ``seq`` is restored rather than reassigned so a replayed decision is
        comparable field-for-field with the recorded one.
        """
        self._seq = event.seq
        result = self._accept(event, origin=event.origin)
        assert result.decision is not None  # _accept always produces one
        return result.decision

    def verify_replay(self, *, config: EngineConfig | None = None) -> VerifyReport:
        """Replay the stored log and assert the decisions match exactly.

        Compares three things, because each can diverge independently:

        * **every decision field** - catches a changed rule outcome;
        * **the final state hash** - catches divergence the labels happened to
          hide;
        * **the audit head hash** - catches a change in the *records* even where
          the decisions themselves agree.
        """
        recorded = [
            record for record in self._audit.records() if record.get("type") != "header"
        ]
        # Duplicates and rejections are audit-only: they never entered the event
        # log, so a replay of that log cannot and should not reproduce them.
        comparable = [
            record for record in recorded
            if record.get("rule") not in {R1_DUPLICATE_EVENT, R2_SCHEMA_REJECT}
        ]

        replayed_engine = self.replay(config=config)
        replayed = [
            record for record in replayed_engine.audit_records()
            if record.get("type") != "header"
        ]

        mismatches: list[Mismatch] = []
        detail = ""

        if len(comparable) != len(replayed):
            detail = (
                f"recorded {len(comparable)} replayable decision(s) but the replay "
                f"produced {len(replayed)}"
            )

        for original, again in zip(comparable, replayed):
            for key in sorted(set(original) | set(again)):
                if key in {"prev_hash", "record_hash"}:
                    continue
                if original.get(key) != again.get(key):
                    mismatches.append(Mismatch(
                        seq=int(original.get("seq", 0)),
                        event_id=str(original.get("event_id", "")),
                        field=key,
                        recorded=original.get(key),
                        replayed=again.get(key),
                    ))

        recorded_state = self.state_hash()
        replayed_state = replayed_engine.state_hash()
        recorded_head = head_hash(list(self._audit.records()))
        replayed_head = head_hash(list(replayed_engine.audit_records()))

        # The audit head only has to match when nothing audit-only was recorded;
        # a run that logged a duplicate legitimately has a longer chain than its
        # replay, and reporting that as tampering would be a false alarm.
        head_comparable = len(comparable) == len(recorded)
        ok = (
            not mismatches
            and not detail
            and recorded_state == replayed_state
            and (not head_comparable or recorded_head == replayed_head)
        )

        return VerifyReport(
            ok=ok,
            events=self.store.event_count(),
            compared=len(replayed),
            mismatches=tuple(mismatches[:100]),
            state_hash_recorded=recorded_state,
            state_hash_replayed=replayed_state,
            audit_head_recorded=recorded_head if head_comparable else "",
            audit_head_replayed=replayed_head if head_comparable else "",
            detail=detail,
        )


# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------

def run_events(
    rows: Iterable[Mapping[str, Any] | RawRow],
    *,
    config: EngineConfig = DEFAULT_CONFIG,
    store: Store | None = None,
    audit: AuditSink | None = None,
) -> tuple[Engine, BatchResult]:
    """Build an engine, ingest ``rows``, and hand back both.

    The shape most tests and the CLI want.
    """
    engine = Engine(config=config, store=store, audit=audit)
    return engine, engine.ingest_rows(rows)
