"""Canonical serialization and hashing - the determinism backbone.

Every guarantee in the PRD's non-functional list (determinism, replayability,
auditability) reduces to one property: *the same logical value must always
produce the same bytes*. That property is implemented exactly once, here, and
everything else in the engine calls into it.

Rules enforced by :func:`canonical_json`:

* keys sorted lexicographically, always;
* no insignificant whitespace (compact separators);
* ``NaN`` / ``Infinity`` rejected outright - they are not valid JSON and they
  are not valid money;
* :class:`~decimal.Decimal`, :class:`~datetime.date` and
  :class:`~datetime.datetime` render through explicit, stable formats rather
  than ``repr``;
* datetimes are always normalized to UTC and rendered with a ``Z`` suffix, so
  ``+00:00`` and ``Z`` inputs cannot produce two different hashes;
* floats are refused. A float in a money path is a determinism bug, so the
  encoder fails loudly instead of silently emitting ``0.30000000000000004``.
"""

from __future__ import annotations

import hashlib
import json
from datetime import date, datetime, timezone
from decimal import Decimal
from typing import Any

# Sentinel used as the previous-hash of the very first audit record in a chain.
GENESIS_HASH = "0" * 64

_COMPACT_SEPARATORS = (",", ":")


def _default(obj: Any) -> Any:
    """Encode types the stdlib encoder does not handle, deterministically."""
    if isinstance(obj, Decimal):
        # Money never reaches JSON as a Decimal in normal operation (we store
        # integer minor units), but diffs and evidence blobs can carry one.
        return format(obj.normalize(), "f")
    if isinstance(obj, datetime):
        return format_ts(obj)
    if isinstance(obj, date):
        return obj.isoformat()
    if isinstance(obj, (set, frozenset)):
        # Sets have no inherent order; sort so the bytes are stable.
        return sorted(obj, key=lambda item: canonical_json(item))
    if isinstance(obj, tuple):
        return list(obj)
    if isinstance(obj, bytes):
        return obj.decode("utf-8", errors="strict")
    raise TypeError(f"{type(obj).__name__} is not canonically serializable: {obj!r}")


class _StrictEncoder(json.JSONEncoder):
    """Rejects floats so they can never enter a hashed payload."""

    def default(self, o: Any) -> Any:  # noqa: D102 - inherited
        return _default(o)


def _reject_floats(value: Any, path: str = "$") -> None:
    """Walk a structure and raise on any float.

    Floats are non-associative and platform-sensitive; allowing one into a
    hashed payload would make ``state_hash`` unreproducible. Ints and bools are
    fine, ``Decimal`` is fine, ``float`` is not.
    """
    if isinstance(value, float):
        raise TypeError(
            f"float found at {path}: {value!r}. Use Decimal or integer minor "
            "units - floats break determinism."
        )
    if isinstance(value, dict):
        for key, item in value.items():
            _reject_floats(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _reject_floats(item, f"{path}[{index}]")


def canonical_json(value: Any, *, allow_floats: bool = False) -> str:
    """Serialize ``value`` to its one canonical JSON representation."""
    if not allow_floats:
        _reject_floats(value)
    return json.dumps(
        value,
        sort_keys=True,
        separators=_COMPACT_SEPARATORS,
        ensure_ascii=False,
        allow_nan=False,
        cls=_StrictEncoder,
    )


def sha256_hex(value: Any) -> str:
    """Canonical-JSON-then-SHA256. The engine's only content hash."""
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def chain_hash(prev_hash: str, record: Any) -> str:
    """Hash a record together with its predecessor.

    Chaining makes the audit log tamper-evident: editing or removing any record
    invalidates every hash after it, which ``zoro audit --verify`` detects.
    """
    payload = canonical_json(record).encode("utf-8")
    return hashlib.sha256(prev_hash.encode("ascii") + b"\x1f" + payload).hexdigest()


def stable_id(*parts: Any, prefix: str = "", length: int = 32) -> str:
    """A deterministic identifier derived from ``parts``.

    Used for synthesized event ids and surrogate transaction ids. The parts are
    canonicalized first so ``("a", 1)`` and ``("a", "1")`` cannot collide.
    """
    digest = hashlib.sha256(canonical_json(list(parts)).encode("utf-8")).hexdigest()
    return f"{prefix}{digest[:length]}" if prefix else digest[:length]


def to_utc(value: datetime) -> datetime:
    """Normalize a datetime to timezone-aware UTC.

    A naive datetime is *assumed* to be UTC rather than rejected. That choice is
    documented in the README: mixed-offset feeds are common in financial data
    and silently reinterpreting them by local timezone would make results
    machine-dependent, which is worse than a documented assumption.
    """
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def format_ts(value: datetime) -> str:
    """Render a datetime as canonical UTC ISO-8601 with a ``Z`` suffix.

    Microseconds are emitted only when non-zero, so the common whole-second
    case stays readable in the audit log.
    """
    utc = to_utc(value)
    if utc.microsecond:
        return utc.strftime("%Y-%m-%dT%H:%M:%S.%f") + "Z"
    return utc.strftime("%Y-%m-%dT%H:%M:%S") + "Z"


def parse_ts(value: str | datetime) -> datetime:
    """Exact inverse of :func:`format_ts`.

    Deliberately strict, unlike :func:`zoro_engine.normalize.parse_timestamp`
    which tolerates whatever a feed sends. This one only ever reads values the
    engine itself wrote (storage columns, audit records), so anything else means
    the data was corrupted or hand-edited - and guessing at it would turn a
    detectable problem into a silently wrong timeline.
    """
    if isinstance(value, datetime):
        return to_utc(value)
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        return to_utc(datetime.fromisoformat(text))
    except ValueError:
        raise ValueError(
            f"{value!r} is not a canonical engine timestamp "
            "(expected YYYY-MM-DDTHH:MM:SS[.ffffff]Z)"
        ) from None


def json_lines(records: list[Any]) -> str:
    """Render records as newline-delimited canonical JSON (always ``\\n``)."""
    if not records:
        return ""
    return "".join(canonical_json(record) + "\n" for record in records)
