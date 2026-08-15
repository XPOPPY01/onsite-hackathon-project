"""Rule-based anomaly detection for unusual transaction edits (bonus scope).

Deliberately **threshold-based, not statistical**. The PRD forbids ML models,
but there is a stronger reason: a detector is only useful in an audit trail if
replaying a two-year-old event produces the same flag it produced originally. A
model whose parameters drift - or that depends on the surrounding population -
cannot offer that. Fixed thresholds can.

For the same reason, "now" is always ``event.event_ts``, never the wall clock.
A detector that asked the system clock how old a transaction is would flag
different things on every replay, and every one of those flags would be
unreproducible.

Flags are advisory. They never change a reconciliation outcome - they are
recorded alongside it so a human can review. Detection that silently altered
resolution would make the rule table in ``docs/RESOLUTION_RULES.md`` a lie.
"""

from __future__ import annotations

from datetime import date as Date
from typing import TYPE_CHECKING, Any

from .config import EngineConfig
from .models import Event
from .money import format_amount

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .state import ApplyOutcome, UserState

# Severity ladder. Ordering matters for display and for the ``--min-severity``
# filter in the CLI.
LOW, MEDIUM, HIGH = "low", "medium", "high"
_SEVERITY_RANK = {LOW: 0, MEDIUM: 1, HIGH: 2}

# Detector codes, exported so tests and dashboards reference one spelling.
A_AMOUNT_JUMP = "A_AMOUNT_JUMP"
A_SIGN_FLIP = "A_SIGN_FLIP"
A_RETRO_DATE_SHIFT = "A_RETRO_DATE_SHIFT"
A_STALE_DELETE = "A_STALE_DELETE"
A_EDIT_CHURN = "A_EDIT_CHURN"
A_ROUND_LARGE_AMOUNT = "A_ROUND_LARGE_AMOUNT"
A_NEAR_DUPLICATE = "A_NEAR_DUPLICATE"
A_CURRENCY_CHANGE = "A_CURRENCY_CHANGE"
A_LOW_AUTHORITY_CONTRADICTION = "A_LOW_AUTHORITY_CONTRADICTION"


def severity_at_least(anomaly: dict[str, Any], minimum: str) -> bool:
    """True if ``anomaly`` is at least as severe as ``minimum``."""
    return _SEVERITY_RANK.get(str(anomaly.get("severity")), 0) >= _SEVERITY_RANK.get(minimum, 0)


def detect_anomalies(
    *,
    event: Event,
    outcome: ApplyOutcome,
    state: UserState,
    config: EngineConfig,
) -> tuple[dict[str, Any], ...]:
    """Run every detector over one applied event.

    Returns a tuple of anomaly dicts, sorted by ``code`` so the audit payload is
    hash-stable regardless of detector execution order.
    """
    if not config.detect_anomalies:
        return ()

    found: list[dict[str, Any]] = []
    thresholds = config.anomaly
    before, after = outcome.before, outcome.after

    _check_amount_change(event, before, after, thresholds, found)
    _check_date_shift(event, before, after, thresholds, found)
    _check_currency_change(event, before, after, found)
    _check_stale_delete(event, outcome, before, thresholds, found)
    _check_edit_churn(event, outcome, thresholds, found)
    _check_round_amount(event, after, thresholds, found)
    _check_near_duplicate(event, outcome, state, found)
    _check_low_authority_contradiction(event, outcome, found)

    return tuple(sorted(found, key=lambda item: (item["code"], item.get("field", ""))))


def _flag(
    found: list[dict[str, Any]],
    code: str,
    severity: str,
    detail: str,
    **evidence: Any,
) -> None:
    entry: dict[str, Any] = {"code": code, "severity": severity, "detail": detail}
    if evidence:
        entry["evidence"] = dict(sorted(evidence.items()))
    found.append(entry)


def _check_amount_change(
    event: Event,
    before: dict[str, Any] | None,
    after: dict[str, Any] | None,
    thresholds: Any,
    found: list[dict[str, Any]],
) -> None:
    """Large absolute or relative amount revisions, and sign flips."""
    if before is None or after is None:
        return
    old = before.get("amount_minor")
    new = after.get("amount_minor")
    if old is None or new is None or old == new:
        return

    currency = after.get("currency") or "USD"
    delta = abs(new - old)
    absolute_hit = delta >= thresholds.amount_jump_minor
    # Guard against division by zero: a change away from exactly zero is treated
    # as an absolute jump only, since "percent of nothing" has no meaning.
    relative_hit = old != 0 and (delta * 100) >= abs(old) * thresholds.amount_jump_pct

    if absolute_hit or relative_hit:
        parts = []
        if absolute_hit:
            parts.append(f"delta {format_amount(delta, currency)} >= "
                         f"{format_amount(thresholds.amount_jump_minor, currency)}")
        if relative_hit:
            parts.append(f"{round(delta * 100 / abs(old))}% >= {thresholds.amount_jump_pct}%")
        _flag(
            found, A_AMOUNT_JUMP,
            HIGH if (absolute_hit and relative_hit) else MEDIUM,
            f"amount revised from {format_amount(old, currency)} to "
            f"{format_amount(new, currency)} ({'; '.join(parts)})",
            field="amount",
            previous=format_amount(old, currency),
            current=format_amount(new, currency),
            source=event.source,
        )

    if thresholds.flag_sign_flip and (old < 0) != (new < 0):
        # A charge becoming a refund (or vice versa) inverts its effect on every
        # rollup it appears in, so it is always worth a look.
        _flag(
            found, A_SIGN_FLIP, HIGH,
            f"amount sign flipped: {format_amount(old, currency)} -> "
            f"{format_amount(new, currency)}",
            field="amount",
            previous=format_amount(old, currency),
            current=format_amount(new, currency),
        )


def _check_date_shift(
    event: Event,
    before: dict[str, Any] | None,
    after: dict[str, Any] | None,
    thresholds: Any,
    found: list[dict[str, Any]],
) -> None:
    """Retroactive date moves large enough to change a reporting period."""
    if before is None or after is None:
        return
    old_raw, new_raw = before.get("date"), after.get("date")
    if not old_raw or not new_raw or old_raw == new_raw:
        return

    old_date, new_date = Date.fromisoformat(old_raw), Date.fromisoformat(new_raw)
    shift = abs((new_date - old_date).days)
    if shift > thresholds.date_shift_days:
        _flag(
            found, A_RETRO_DATE_SHIFT,
            HIGH if shift > thresholds.date_shift_days * 4 else MEDIUM,
            f"transaction date moved {shift} days ({old_raw} -> {new_raw}), "
            f"over the {thresholds.date_shift_days}-day threshold",
            field="date", previous=old_raw, current=new_raw, shift_days=shift,
            source=event.source,
        )


def _check_currency_change(
    event: Event,
    before: dict[str, Any] | None,
    after: dict[str, Any] | None,
    found: list[dict[str, Any]],
) -> None:
    """A currency change reinterprets the stored amount - always notable."""
    if before is None or after is None:
        return
    old, new = before.get("currency"), after.get("currency")
    if old and new and old != new:
        _flag(
            found, A_CURRENCY_CHANGE, HIGH,
            f"currency changed {old} -> {new}; amount was not converted",
            field="currency", previous=old, current=new, source=event.source,
        )


def _check_stale_delete(
    event: Event,
    outcome: ApplyOutcome,
    before: dict[str, Any] | None,
    thresholds: Any,
    found: list[dict[str, Any]],
) -> None:
    """Deletion of a long-settled transaction."""
    if event.action != "delete" or outcome.state_diff.get("op") != "deleted":
        return
    raw_date = (before or {}).get("date")
    if not raw_date:
        return
    age = (event.event_ts.date() - Date.fromisoformat(raw_date)).days
    if age > thresholds.delete_age_days:
        _flag(
            found, A_STALE_DELETE, MEDIUM,
            f"deleted a transaction dated {raw_date}, {age} days before the edit "
            f"(threshold {thresholds.delete_age_days})",
            field="deleted", transaction_date=raw_date, age_days=age, source=event.source,
        )


def _check_edit_churn(
    event: Event,
    outcome: ApplyOutcome,
    thresholds: Any,
    found: list[dict[str, Any]],
) -> None:
    """An unusual number of accepted edits against one transaction."""
    count = len(outcome.considered)
    if count > thresholds.edit_churn_count:
        _flag(
            found, A_EDIT_CHURN, LOW if count <= thresholds.edit_churn_count * 2 else MEDIUM,
            f"{count} events now recorded against this transaction "
            f"(threshold {thresholds.edit_churn_count})",
            event_count=count,
            sources=sorted({item.source for item in outcome.considered}),
        )


def _check_round_amount(
    event: Event,
    after: dict[str, Any] | None,
    thresholds: Any,
    found: list[dict[str, Any]],
) -> None:
    """Large, suspiciously round amounts - a classic manual-entry tell."""
    if after is None or event.action == "delete":
        return
    amount = after.get("amount_minor")
    if amount is None:
        return
    magnitude = abs(amount)
    if (
        magnitude >= thresholds.round_amount_floor_minor
        and thresholds.round_amount_modulus_minor > 0
        and magnitude % thresholds.round_amount_modulus_minor == 0
    ):
        currency = after.get("currency") or "USD"
        _flag(
            found, A_ROUND_LARGE_AMOUNT, LOW,
            f"large round amount {format_amount(amount, currency)} "
            f"(multiple of {format_amount(thresholds.round_amount_modulus_minor, currency)})",
            field="amount", amount=format_amount(amount, currency), source=event.source,
        )


def _check_near_duplicate(
    event: Event,
    outcome: ApplyOutcome,
    state: UserState,
    found: list[dict[str, Any]],
) -> None:
    """A different transaction id with an identical date and amount.

    The duplicate the id-based rules cannot catch: the same purchase imported
    twice under two different references. Flagged rather than merged, because
    two genuinely separate identical purchases on one day are entirely possible
    and auto-merging them would destroy real data.
    """
    txn = outcome.transaction
    if txn is None or txn.deleted or txn.amount_minor is None or txn.date is None:
        return

    matches = [
        other.transaction_id
        for _key, other in sorted(state.transactions.items())
        if other.transaction_key != txn.transaction_key
        and not other.deleted
        and other.amount_minor == txn.amount_minor
        and other.date == txn.date
        and other.currency == txn.currency
    ]
    if matches:
        _flag(
            found, A_NEAR_DUPLICATE, MEDIUM,
            f"{len(matches)} other live transaction(s) share this date and amount "
            "under a different id; possible unlinked duplicate",
            amount=txn.amount, date=txn.date.isoformat(),
            matches=matches[:10], match_count=len(matches),
        )


def _check_low_authority_contradiction(
    event: Event,
    outcome: ApplyOutcome,
    found: list[dict[str, Any]],
) -> None:
    """A low-authority feed repeatedly contradicting a higher-authority value.

    Not a resolution problem - the engine already handled it correctly - but a
    strong signal that a feed is misconfigured or fighting a user's corrections.
    """
    effect = outcome.effect
    if not effect.lost or effect.won:
        return
    if event.source == "user_edit":
        return
    _flag(
        found, A_LOW_AUTHORITY_CONTRADICTION, LOW,
        f"{event.source} asserted {', '.join(effect.lost)} but was outranked; "
        "the feed disagrees with a higher-authority value",
        fields=list(effect.lost), source=event.source, authority=event.authority,
    )
