"""Engine configuration: the authority model, rule switches and thresholds.

Every value that a reconciliation decision depends on lives in
:class:`EngineConfig`. That is a deliberate design constraint, not a style
preference - the PRD requires that identical inputs produce identical outputs,
and a decision is only reproducible if the *rules* it ran under are pinned too.

:meth:`EngineConfig.fingerprint` hashes the whole configuration and the engine
stamps that fingerprint into the audit log header. Replaying a log under a
different fingerprint is therefore detectable rather than silently wrong.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from datetime import date
from typing import Any, Mapping

from .canonical import sha256_hex
from .errors import ConfigError
from .money import DEFAULT_CURRENCY

# ---------------------------------------------------------------------------
# Decision vocabulary
# ---------------------------------------------------------------------------

#: The four labels the PRD mandates in the ``decision`` field.
PRD_DECISIONS: tuple[str, ...] = ("merged", "replaced", "deleted", "ignored")

#: Two additions. ``created`` because forcing a first-time insert into "merged"
#: would make the audit trail actively misleading, and ``rejected`` because a
#: schema failure is not a reconciliation outcome - nothing was ignored, the
#: row never entered the timeline at all.
EXTRA_DECISIONS: tuple[str, ...] = ("created", "rejected")

DECISIONS: frozenset[str] = frozenset(PRD_DECISIONS + EXTRA_DECISIONS)


# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------

CANONICAL_ACTIONS: tuple[str, ...] = ("create", "update", "delete")

#: Real feeds spell the same intent many ways. Normalizing aliases at the
#: boundary keeps the resolver working on three cases instead of thirty.
ACTION_ALIASES: dict[str, str] = {
    "create": "create", "created": "create", "insert": "create", "new": "create",
    "add": "create", "added": "create", "c": "create", "post": "create",
    "update": "update", "updated": "update", "edit": "update", "edited": "update",
    "modify": "update", "modified": "update", "change": "update", "correct": "update",
    "correction": "update", "patch": "update", "amend": "update", "u": "update",
    "delete": "delete", "deleted": "delete", "remove": "delete", "removed": "delete",
    "del": "delete", "d": "delete", "void": "delete", "voided": "delete",
    "cancel": "delete", "cancelled": "delete", "canceled": "delete", "reverse": "delete",
}

#: Tie-break order when two events on one transaction share an ``event_ts``.
#: ``delete`` sorts last so that a same-instant create/delete pair settles as
#: deleted: destructive intent is the safer resolution when the feed gives us
#: no way to order the two.
ACTION_RANK: dict[str, int] = {"create": 0, "update": 1, "delete": 2}


# ---------------------------------------------------------------------------
# Sources and authority
# ---------------------------------------------------------------------------

CANONICAL_SOURCES: tuple[str, ...] = ("user_edit", "third_party", "bank_sync", "unknown")

SOURCE_ALIASES: dict[str, str] = {
    # A human deliberately asserting a value - the highest authority we have.
    "user_edit": "user_edit", "user": "user_edit", "user_correction": "user_edit",
    "manual": "user_edit", "manual_entry": "user_edit", "manual_input": "user_edit",
    "app": "user_edit", "app_edit": "user_edit", "mobile": "user_edit",
    "web": "user_edit", "ui": "user_edit", "human": "user_edit",
    # Aggregators and importers: structured, but a step removed from both the
    # bank's ledger and the user's intent.
    "third_party": "third_party", "thirdparty": "third_party", "3p": "third_party",
    "expense_tracker": "third_party", "tracker": "third_party", "partner": "third_party",
    "csv_import": "third_party", "csv": "third_party", "import": "third_party",
    "integration": "third_party", "spreadsheet": "third_party", "excel": "third_party",
    # Authoritative on settlement, but routinely wrong about the things users
    # care about (merchant names, categories, pending-vs-posted dates).
    "bank_sync": "bank_sync", "bank": "bank_sync", "bank_api": "bank_sync",
    "sync": "bank_sync", "aggregator": "bank_sync", "plaid": "bank_sync",
    "open_banking": "bank_sync", "feed": "bank_sync", "statement": "bank_sync",
}

#: Higher wins. Gaps are intentional so operators can slot custom sources
#: between the built-in tiers without renumbering everything.
DEFAULT_SOURCE_PRIORITY: dict[str, int] = {
    "user_edit": 100,
    "third_party": 50,
    "bank_sync": 10,
    "unknown": 1,
}

#: Bonus scope: role-based reconciliation priority. A premium user's manual
#: correction outranks a standard user's.
DEFAULT_ROLE_BONUS: dict[str, int] = {
    "admin": 25,
    "premium": 10,
    "standard": 0,
    "trial": -5,
}

#: The role bonus applies only to sources that represent *the user speaking*.
#: A premium subscription does not make their bank's feed more accurate, so
#: widening this would be a modelling error rather than a generosity.
ROLE_BONUS_SOURCES: tuple[str, ...] = ("user_edit",)


# ---------------------------------------------------------------------------
# Transaction shape
# ---------------------------------------------------------------------------

#: Fields the resolver reconciles field-by-field. Order is fixed because it
#: determines the key order of ``state_diff`` payloads, which are hashed.
MUTABLE_FIELDS: tuple[str, ...] = (
    "amount_minor",
    "currency",
    "date",
    "category",
    "merchant",
    "account_id",
)

#: Fields where an empty incoming value must never overwrite a known one.
#: Bank feeds habitually send blank categories; treating that as "the user
#: cleared it" would destroy data on every sync.
TEXT_FIELDS: tuple[str, ...] = ("category", "merchant", "account_id")


@dataclass(frozen=True)
class SourcePriority:
    """Resolves an authority score for a ``(source, role)`` pair.

    Authority - not recency - is the primary axis of conflict resolution. The
    PRD's "prefer user edits over bank syncs" is unqualified, so a later bank
    sync must not silently revert an earlier user correction.
    """

    base: Mapping[str, int] = field(default_factory=lambda: dict(DEFAULT_SOURCE_PRIORITY))
    role_bonus: Mapping[str, int] = field(default_factory=lambda: dict(DEFAULT_ROLE_BONUS))
    unknown_priority: int = 1
    bonus_sources: tuple[str, ...] = ROLE_BONUS_SOURCES

    def score(self, source: str, role: str | None = None) -> int:
        """Authority score for an event from ``source`` by a ``role`` user."""
        base = self.base.get(source, self.unknown_priority)
        if role and source in self.bonus_sources:
            base += self.role_bonus.get(role, 0)
        return base

    def to_dict(self) -> dict[str, Any]:
        return {
            "base": dict(sorted(self.base.items())),
            "role_bonus": dict(sorted(self.role_bonus.items())),
            "unknown_priority": self.unknown_priority,
            "bonus_sources": list(self.bonus_sources),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> SourcePriority:
        return cls(
            base=dict(payload.get("base", DEFAULT_SOURCE_PRIORITY)),
            role_bonus=dict(payload.get("role_bonus", DEFAULT_ROLE_BONUS)),
            unknown_priority=int(payload.get("unknown_priority", 1)),
            bonus_sources=tuple(payload.get("bonus_sources", ROLE_BONUS_SOURCES)),
        )


@dataclass(frozen=True)
class LockedPeriod:
    """A closed accounting period. Deletes inside it are refused.

    This is the concrete meaning the engine gives to the PRD's "remove
    transaction if no dependencies exist": a booked period is a dependency,
    because reports have already been filed against it.
    """

    start: date
    end: date
    label: str = ""

    def contains(self, value: date) -> bool:
        return self.start <= value <= self.end

    def to_dict(self) -> dict[str, Any]:
        return {"start": self.start.isoformat(), "end": self.end.isoformat(), "label": self.label}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> LockedPeriod:
        return cls(
            start=date.fromisoformat(str(payload["start"])),
            end=date.fromisoformat(str(payload["end"])),
            label=str(payload.get("label", "")),
        )


# ---------------------------------------------------------------------------
# Anomaly detection thresholds (bonus scope, rule-based - no ML)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AnomalyThresholds:
    """Fixed thresholds for flagging unusual edits.

    Deliberately rule-based: the PRD forbids ML/LLM models, and fixed
    thresholds are also the only kind of detector that can be replayed to an
    identical result years later.
    """

    #: Absolute jump, in minor units, that makes an amount edit notable.
    amount_jump_minor: int = 100_000          # 1,000.00
    #: ...or a relative jump, as a percentage of the previous amount.
    amount_jump_pct: int = 300
    #: A retroactive date move larger than this many days is suspicious.
    date_shift_days: int = 90
    #: Deleting a transaction older than this is suspicious.
    delete_age_days: int = 365
    #: More than this many accepted edits on one transaction is churn.
    edit_churn_count: int = 5
    #: Large, suspiciously round amounts (in minor units).
    round_amount_floor_minor: int = 1_000_000  # 10,000.00
    round_amount_modulus_minor: int = 100_000  # multiples of 1,000.00
    #: Sign flips (refund <-> charge) are always worth a flag.
    flag_sign_flip: bool = True

    def to_dict(self) -> dict[str, Any]:
        return dict(sorted(asdict(self).items()))

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> AnomalyThresholds:
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in payload.items() if k in known})


# ---------------------------------------------------------------------------
# Engine configuration
# ---------------------------------------------------------------------------

DEFAULT_MAX_UPLOAD_BYTES = 10 * 1024 * 1024   # PRD: CSV uploads capped at 10MB
MIN_PLAUSIBLE_DATE = date(1970, 1, 1)
MAX_PLAUSIBLE_DATE = date(2100, 12, 31)


@dataclass(frozen=True)
class EngineConfig:
    """Immutable, hashable configuration for one engine instance."""

    # --- authority model ---------------------------------------------------
    priority: SourcePriority = field(default_factory=SourcePriority)
    user_roles: Mapping[str, str] = field(default_factory=dict)
    default_role: str = "standard"

    # --- resolution rule switches ----------------------------------------
    #: R4/R5: on *equal* authority, an earlier date wins. This is the PRD's
    #: "merge with existing record if date is earlier" / "merging with earliest
    #: timestamp". A strictly higher authority can still move a date later,
    #: otherwise a user could never fix a too-early bank posting date - which
    #: would contradict "prefer user edits". See docs/RESOLUTION_RULES.md.
    prefer_earliest_date_on_equal_authority: bool = True
    #: R6: an update arriving before its create provisionally materializes the
    #: transaction rather than dropping the data on the floor.
    allow_update_before_create: bool = True
    #: R9: a delete arriving before its create records a tombstone, so the
    #: late create cannot silently resurrect a deleted row.
    allow_delete_before_create: bool = True
    #: R7: an update from a strictly higher authority than the delete undoes
    #: the delete (a user editing a row is asserting it exists).
    resurrect_deleted_on_higher_authority: bool = True
    #: Blank incoming text must not overwrite a populated field.
    blank_never_overwrites: bool = True

    # --- dependency guards ------------------------------------------------
    enforce_delete_dependencies: bool = True
    locked_periods: Mapping[str, tuple[LockedPeriod, ...]] = field(default_factory=dict)

    # --- ingest / normalization -------------------------------------------
    default_currency: str = DEFAULT_CURRENCY
    #: Reject ``03/04/2024`` rather than guess between March 4 and April 3.
    #: Guessing would be deterministic but wrong half the time; refusing is
    #: deterministic and honest.
    strict_ambiguous_dates: bool = True
    #: Treat ``TXN-1`` and ``txn-1`` as the same transaction.
    case_insensitive_ids: bool = True
    #: Derive a surrogate id from the payload when ``transaction_id`` is blank.
    synthesize_missing_transaction_id: bool = True
    #: Fall back to ``date T00:00:00Z`` when ``event_ts`` is absent.
    infer_missing_event_ts: bool = True
    min_date: date = MIN_PLAUSIBLE_DATE
    max_date: date = MAX_PLAUSIBLE_DATE
    max_upload_bytes: int = DEFAULT_MAX_UPLOAD_BYTES

    # --- bonus features ---------------------------------------------------
    detect_anomalies: bool = True
    anomaly: AnomalyThresholds = field(default_factory=AnomalyThresholds)

    def __post_init__(self) -> None:
        if self.default_role not in self.priority.role_bonus:
            raise ConfigError(
                f"default_role {self.default_role!r} has no entry in role_bonus "
                f"({sorted(self.priority.role_bonus)})"
            )
        if self.min_date > self.max_date:
            raise ConfigError(f"min_date {self.min_date} is after max_date {self.max_date}")
        if self.max_upload_bytes <= 0:
            raise ConfigError(f"max_upload_bytes must be positive, got {self.max_upload_bytes}")

    # --- authority helpers ------------------------------------------------

    def role_for(self, user_id: str) -> str:
        """Role of ``user_id``, falling back to :attr:`default_role`."""
        return self.user_roles.get(user_id, self.default_role)

    def authority(self, source: str, user_id: str) -> int:
        """Authority score for an event from ``source`` on behalf of ``user_id``."""
        return self.priority.score(source, self.role_for(user_id))

    def is_locked(self, user_id: str, value: date | None) -> LockedPeriod | None:
        """Return the locked period covering ``value``, if any.

        ``"*"`` is honoured as a wildcard key for platform-wide period locks.
        """
        if value is None or not self.enforce_delete_dependencies:
            return None
        for key in (user_id, "*"):
            for period in self.locked_periods.get(key, ()):
                if period.contains(value):
                    return period
        return None

    # --- serialization ----------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Canonical, JSON-safe view. Feeds :meth:`fingerprint`."""
        return {
            "priority": self.priority.to_dict(),
            "user_roles": dict(sorted(self.user_roles.items())),
            "default_role": self.default_role,
            "prefer_earliest_date_on_equal_authority": self.prefer_earliest_date_on_equal_authority,
            "allow_update_before_create": self.allow_update_before_create,
            "allow_delete_before_create": self.allow_delete_before_create,
            "resurrect_deleted_on_higher_authority": self.resurrect_deleted_on_higher_authority,
            "blank_never_overwrites": self.blank_never_overwrites,
            "enforce_delete_dependencies": self.enforce_delete_dependencies,
            "locked_periods": {
                user: [period.to_dict() for period in periods]
                for user, periods in sorted(self.locked_periods.items())
            },
            "default_currency": self.default_currency,
            "strict_ambiguous_dates": self.strict_ambiguous_dates,
            "case_insensitive_ids": self.case_insensitive_ids,
            "synthesize_missing_transaction_id": self.synthesize_missing_transaction_id,
            "infer_missing_event_ts": self.infer_missing_event_ts,
            "min_date": self.min_date.isoformat(),
            "max_date": self.max_date.isoformat(),
            "max_upload_bytes": self.max_upload_bytes,
            "detect_anomalies": self.detect_anomalies,
            "anomaly": self.anomaly.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> EngineConfig:
        """Rebuild a config from :meth:`to_dict` output.

        Round-trips exactly, so a stored config can be reloaded to replay a
        historical log under the rules that produced it.
        """
        data = dict(payload)
        kwargs: dict[str, Any] = {}

        if "priority" in data:
            kwargs["priority"] = SourcePriority.from_dict(data["priority"])
        if "anomaly" in data:
            kwargs["anomaly"] = AnomalyThresholds.from_dict(data["anomaly"])
        if "locked_periods" in data:
            kwargs["locked_periods"] = {
                user: tuple(LockedPeriod.from_dict(p) for p in periods)
                for user, periods in data["locked_periods"].items()
            }
        for key in ("min_date", "max_date"):
            if key in data:
                kwargs[key] = date.fromisoformat(str(data[key]))

        passthrough = {
            "user_roles", "default_role", "prefer_earliest_date_on_equal_authority",
            "allow_update_before_create", "allow_delete_before_create",
            "resurrect_deleted_on_higher_authority", "blank_never_overwrites",
            "enforce_delete_dependencies", "default_currency", "strict_ambiguous_dates",
            "case_insensitive_ids", "synthesize_missing_transaction_id",
            "infer_missing_event_ts", "max_upload_bytes", "detect_anomalies",
        }
        for key in passthrough:
            if key in data:
                kwargs[key] = data[key]

        if "user_roles" in kwargs:
            kwargs["user_roles"] = dict(kwargs["user_roles"])

        return cls(**kwargs)

    def with_overrides(self, **overrides: Any) -> EngineConfig:
        """A copy with ``overrides`` applied (configs are immutable)."""
        return replace(self, **overrides)

    def fingerprint(self) -> str:
        """Stable hash of the full rule set.

        Stamped into the audit log header. A replay under a different
        fingerprint is a different experiment, and the engine says so instead
        of quietly producing different decisions.
        """
        return sha256_hex(self.to_dict())


#: Shared default. Immutable, so handing the same instance to every engine is
#: safe and keeps fingerprints comparable across runs.
DEFAULT_CONFIG = EngineConfig()
