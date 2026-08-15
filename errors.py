"""Exception hierarchy for the reconciliation engine.

Every error carries a stable machine-readable ``code`` so that API responses,
CLI output and the audit trail can all reference the same identifier. Codes are
part of the contract: tests assert on them and the audit log stores them.
"""

from __future__ import annotations

from typing import Any


class ZoroError(Exception):
    """Base class for every error raised by the engine."""

    code = "E_ZORO"

    def __init__(self, message: str, *, code: str | None = None, **context: Any) -> None:
        super().__init__(message)
        self.message = message
        if code is not None:
            self.code = code
        self.context: dict[str, Any] = context

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"code": self.code, "message": self.message}
        if self.context:
            payload["context"] = self.context
        return payload

    def __str__(self) -> str:  # pragma: no cover - trivial
        return f"[{self.code}] {self.message}"


class ValidationError(ZoroError):
    """A row or payload failed schema validation / normalization.

    Maps to HTTP 400. ``issues`` holds the full list of per-field problems so
    the caller can fix an entire row in one pass instead of one field per
    round-trip.
    """

    code = "E_VALIDATION"

    def __init__(
        self,
        message: str,
        *,
        issues: list[dict[str, Any]] | None = None,
        **context: Any,
    ) -> None:
        super().__init__(message, **context)
        self.issues = issues or []

    def to_dict(self) -> dict[str, Any]:
        payload = super().to_dict()
        payload["issues"] = self.issues
        return payload


class DuplicateEventError(ZoroError):
    """The event_id has already been processed. Maps to HTTP 409."""

    code = "E_DUPLICATE_EVENT"

    def __init__(self, message: str, *, event_id: str, **context: Any) -> None:
        super().__init__(message, event_id=event_id, **context)
        self.event_id = event_id


class IngestError(ZoroError):
    """The payload could not be read at all (bad CSV, oversized upload...)."""

    code = "E_INGEST"


class StoreError(ZoroError):
    """The persistence layer refused an operation."""

    code = "E_STORE"


class ReplayMismatchError(ZoroError):
    """A replay did not reproduce the original decisions byte-for-byte.

    This is the loudest failure the engine can produce: it means determinism
    has been violated, so the audit trail can no longer be trusted.
    """

    code = "E_REPLAY_MISMATCH"

    def __init__(self, message: str, *, mismatches: list[dict[str, Any]], **context: Any) -> None:
        super().__init__(message, **context)
        self.mismatches = mismatches

    def to_dict(self) -> dict[str, Any]:
        payload = super().to_dict()
        payload["mismatches"] = self.mismatches
        return payload


class ConfigError(ZoroError):
    """Engine configuration is internally inconsistent."""

    code = "E_CONFIG"
