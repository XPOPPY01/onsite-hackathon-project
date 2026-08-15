"""The audit trail: a tamper-evident, byte-reproducible decision log.

``audit.log`` is newline-delimited canonical JSON. Two properties are load-
bearing:

**It is byte-for-byte reproducible.** No wall-clock timestamp appears anywhere
in it. ``timestamp`` is the *event's* time, not the ingest time, so re-running
the same events under the same config produces an identical file - which is what
makes ``zoro verify`` a real check rather than a formality. Wall-clock run
metadata lives in ``run_manifest.json``, deliberately outside the hashed log.

**It is a hash chain.** Each record carries ``prev_hash`` and ``record_hash``.
Editing, reordering or deleting any record invalidates every hash after it, and
:func:`verify_chain` reports the first break.

The honest limit of that guarantee: an attacker who rewrites *every* record from
the tampered point to the end produces a chain that verifies internally. What a
hash chain buys is that tampering cannot be **local** and cannot be **silent** -
the head hash necessarily changes. :func:`head_hash` returns that value so it can
be anchored somewhere the attacker does not control (printed in CI output,
committed, or stored beside the run manifest); comparing it is what upgrades
"internally consistent" to "actually unmodified".

The chain is seeded by a **header** record holding the engine version and the
config fingerprint. Because the header is part of the chain, a log produced under
different resolution rules cannot be confused with one produced under the
current rules - the hashes diverge from the very first record.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Protocol, Sequence

from .canonical import GENESIS_HASH, canonical_json, chain_hash
from .config import EngineConfig
from .errors import StoreError
from .models import Decision, seal_decision

#: Bumped when the record layout changes incompatibly.
AUDIT_SCHEMA_VERSION = 1

DEFAULT_AUDIT_FILENAME = "audit.log"


def build_header(config: EngineConfig, *, engine_version: str) -> dict[str, Any]:
    """The chain-seeding header record.

    Contains no timestamp on purpose - see the module docstring.
    """
    return {
        "type": "header",
        "schema_version": AUDIT_SCHEMA_VERSION,
        "engine_version": engine_version,
        "config_fingerprint": config.fingerprint(),
    }


# ---------------------------------------------------------------------------
# Sinks
# ---------------------------------------------------------------------------

class AuditSink(Protocol):
    """Destination for audit records.

    Implementations own the chain: :meth:`append` seals a decision against
    :attr:`last_hash` and returns the sealed copy, so callers cannot accidentally
    write an unchained record.
    """

    @property
    def last_hash(self) -> str: ...

    def start(self, header: dict[str, Any]) -> None: ...

    def append(self, decision: Decision) -> Decision: ...

    def records(self) -> Sequence[dict[str, Any]]: ...

    def close(self) -> None: ...


class _BaseSink:
    """Shared chain bookkeeping."""

    def __init__(self) -> None:
        self._last_hash = GENESIS_HASH
        self._started = False

    @property
    def last_hash(self) -> str:
        return self._last_hash

    def _seal_header(self, header: dict[str, Any]) -> dict[str, Any]:
        record = dict(header)
        record["prev_hash"] = GENESIS_HASH
        record["record_hash"] = chain_hash(GENESIS_HASH, header)
        self._last_hash = record["record_hash"]
        self._started = True
        return record

    def _seal(self, decision: Decision) -> Decision:
        sealed = seal_decision(decision, self._last_hash)
        self._last_hash = sealed.record_hash
        return sealed


class MemoryAuditSink(_BaseSink):
    """In-memory sink. Used for replay, tests and dry runs.

    Replay must never touch the live ``audit.log`` - reproducing a decision is a
    read-only act, and writing replayed records into the real trail would corrupt
    the very history being verified.
    """

    def __init__(self) -> None:
        super().__init__()
        self._records: list[dict[str, Any]] = []

    def start(self, header: dict[str, Any]) -> None:
        if self._started:
            return
        self._records.append(self._seal_header(header))

    def append(self, decision: Decision) -> Decision:
        sealed = self._seal(decision)
        self._records.append(sealed.to_audit_record())
        return sealed

    def records(self) -> Sequence[dict[str, Any]]:
        return tuple(self._records)

    def close(self) -> None:
        return None

    def render(self) -> str:
        """The log as it would appear on disk."""
        return "".join(canonical_json(record) + "\n" for record in self._records)


class FileAuditSink(_BaseSink):
    """Appends to ``audit.log`` on disk.

    Opened in append mode by default so a log survives across runs; the existing
    tail is read first so the chain continues rather than restarting. Every write
    is flushed, because an audit trail that loses its last records in a crash is
    not an audit trail.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        truncate: bool = False,
        flush: bool = True,
    ) -> None:
        super().__init__()
        self.path = Path(path)
        self._flush = flush
        self.path.parent.mkdir(parents=True, exist_ok=True)

        if truncate and self.path.exists():
            self.path.unlink()

        existing = read_audit(self.path) if self.path.exists() else []
        if existing:
            self._last_hash = str(existing[-1].get("record_hash") or GENESIS_HASH)
            self._started = True

        # newline="\n" is essential: the default would translate to CRLF on
        # Windows and the log would stop being byte-identical across platforms.
        self._handle = self.path.open("a", encoding="utf-8", newline="\n")

    def start(self, header: dict[str, Any]) -> None:
        if self._started:
            return
        self._write(self._seal_header(header))

    def append(self, decision: Decision) -> Decision:
        sealed = self._seal(decision)
        self._write(sealed.to_audit_record())
        return sealed

    def _write(self, record: dict[str, Any]) -> None:
        self._handle.write(canonical_json(record) + "\n")
        if self._flush:
            self._handle.flush()
            os.fsync(self._handle.fileno())

    def records(self) -> Sequence[dict[str, Any]]:
        self._handle.flush()
        return tuple(read_audit(self.path))

    def close(self) -> None:
        if not self._handle.closed:
            self._handle.flush()
            self._handle.close()

    def __enter__(self) -> FileAuditSink:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


class NullAuditSink(_BaseSink):
    """Discards records but still maintains the chain.

    Keeps ``record_hash`` values meaningful during benchmarks, so a performance
    run measures the same work the real pipeline does.
    """

    def start(self, header: dict[str, Any]) -> None:
        if not self._started:
            self._seal_header(header)

    def append(self, decision: Decision) -> Decision:
        return self._seal(decision)

    def records(self) -> Sequence[dict[str, Any]]:
        return ()

    def close(self) -> None:
        return None


# ---------------------------------------------------------------------------
# Reading and verification
# ---------------------------------------------------------------------------

def read_audit(path: str | Path) -> list[dict[str, Any]]:
    """Read an audit log into records, reporting the line number on failure."""
    target = Path(path)
    if not target.exists():
        raise StoreError(f"audit log {target} does not exist", path=str(target))

    records: list[dict[str, Any]] = []
    with target.open("r", encoding="utf-8", newline="") as handle:
        for lineno, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                records.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                raise StoreError(
                    f"{target}:{lineno} is not valid JSON: {exc.msg}",
                    path=str(target), line=lineno,
                ) from None
    return records


def iter_decisions(records: Iterable[dict[str, Any]]) -> Iterator[dict[str, Any]]:
    """Yield only decision records, skipping the header."""
    for record in records:
        if record.get("type") != "header":
            yield record


@dataclass(frozen=True, slots=True)
class ChainReport:
    """Result of verifying an audit chain."""

    ok: bool
    checked: int
    broken_at: int | None = None
    detail: str = ""
    expected: str = ""
    found: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"ok": self.ok, "records_checked": self.checked}
        if not self.ok:
            payload.update({
                "broken_at": self.broken_at,
                "detail": self.detail,
                "expected_hash": self.expected,
                "found_hash": self.found,
            })
        return payload

    def __str__(self) -> str:  # pragma: no cover - display only
        if self.ok:
            return f"audit chain intact across {self.checked} record(s)"
        return f"audit chain broken at record {self.broken_at}: {self.detail}"


def verify_chain(records: Sequence[dict[str, Any]]) -> ChainReport:
    """Recompute every hash and report the first divergence.

    Detects any *local* edit: a changed field, a deleted record, a reordering, or
    a record whose ``record_hash`` was recomputed to cover an edit (the link to
    the following record then breaks). It cannot detect a rewrite of the entire
    tail - see the module docstring and :func:`head_hash`.
    """
    expected_prev = GENESIS_HASH

    for index, record in enumerate(records):
        body = {
            key: value for key, value in record.items()
            if key not in {"prev_hash", "record_hash"}
        }
        stored_prev = str(record.get("prev_hash", ""))
        stored_hash = str(record.get("record_hash", ""))

        if stored_prev != expected_prev:
            return ChainReport(
                ok=False, checked=index, broken_at=index,
                detail=f"prev_hash does not match the previous record's hash "
                       f"(event_id={record.get('event_id', record.get('type'))})",
                expected=expected_prev, found=stored_prev,
            )

        recomputed = chain_hash(stored_prev, body)
        if recomputed != stored_hash:
            return ChainReport(
                ok=False, checked=index, broken_at=index,
                detail=f"record_hash does not match its contents - the record was "
                       f"modified after writing "
                       f"(event_id={record.get('event_id', record.get('type'))})",
                expected=recomputed, found=stored_hash,
            )

        expected_prev = stored_hash

    return ChainReport(ok=True, checked=len(records))


def verify_audit_file(path: str | Path) -> ChainReport:
    """Convenience wrapper: read a log and verify its chain."""
    return verify_chain(read_audit(path))


def head_hash(records: Sequence[dict[str, Any]]) -> str:
    """The last record's hash - the single value that commits to the whole log.

    Anchor this outside the log itself (CI output, a commit, the run manifest) and
    compare it on later verification. Without an external anchor a hash chain can
    only prove internal consistency; with one it proves the log is unmodified.
    """
    if not records:
        return GENESIS_HASH
    return str(records[-1].get("record_hash") or GENESIS_HASH)


def summarize(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Counts by decision, rule and reason - the shape a reviewer scans first."""
    decisions: dict[str, int] = {}
    rules: dict[str, int] = {}
    reasons: dict[str, int] = {}
    users: set[str] = set()
    conflicts = 0
    anomalies = 0
    total = 0

    for record in iter_decisions(records):
        total += 1
        decisions[str(record.get("decision"))] = decisions.get(str(record.get("decision")), 0) + 1
        rules[str(record.get("rule"))] = rules.get(str(record.get("rule")), 0) + 1
        for code in str(record.get("reason", "")).split(";"):
            if code:
                # Strip the ``key:value`` detail so counts group by reason kind.
                head = code.split(":", 1)[0]
                reasons[head] = reasons.get(head, 0) + 1
        if record.get("user_id"):
            users.add(str(record["user_id"]))
        conflicts += len(record.get("conflicts") or ())
        anomalies += len(record.get("anomalies") or ())

    return {
        "events": total,
        "users": len(users),
        "by_decision": dict(sorted(decisions.items())),
        "by_rule": dict(sorted(rules.items())),
        "by_reason": dict(sorted(reasons.items())),
        "conflicts": conflicts,
        "anomalies": anomalies,
    }
