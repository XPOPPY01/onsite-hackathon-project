"""Row normalization and schema validation - the engine's boundary layer.

Everything messy about real financial feeds is handled here so that the
resolver downstream only ever sees exact, well-typed values: header aliases,
currency symbols, ``"N/A"`` placeholders, mixed date formats, alias spellings
of the three actions, and missing identifiers.

Two decisions worth calling out, because both trade convenience for
correctness:

**Ambiguous dates are rejected, not guessed.** ``03/04/2024`` is March 4th in
the US and April 3rd nearly everywhere else. Picking one would be deterministic
but silently wrong about half the time, and a wrong transaction date corrupts
every budget rollup it lands in. :data:`EngineConfig.strict_ambiguous_dates`
refuses the row and names the fix. Formats that *cannot* be misread - ISO,
``05-Mar-2024``, ``25/03/2024`` where 25 cannot be a month - are accepted.

**A row with neither ``event_ts`` nor ``date`` is rejected.** Reconciliation is
the act of ordering competing assertions in time; a row with no clock cannot be
placed in a timeline. The alternative - falling back to arrival order - would
quietly make the final state depend on upload order and break the PRD's
determinism guarantee for exactly the rows most likely to be in conflict.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from datetime import date as Date
from datetime import datetime, timezone
from typing import Any, Mapping

from .canonical import format_ts, stable_id, to_utc
from .config import ACTION_ALIASES, DEFAULT_CONFIG, EngineConfig, SOURCE_ALIASES
from .errors import ValidationError
from .models import Event, Issue
from .money import normalize_currency, parse_amount

# ---------------------------------------------------------------------------
# Input shape tolerance
# ---------------------------------------------------------------------------

#: Header aliases. Feeds and hand-edited spreadsheets rarely use the exact
#: field names from the spec, and rejecting a whole file over a column called
#: ``txn_id`` would make the tool useless in practice.
COLUMN_ALIASES: dict[str, str] = {
    "user_id": "user_id", "user": "user_id", "uid": "user_id", "userid": "user_id",
    "customer_id": "user_id", "customer": "user_id", "account_holder": "user_id",

    "transaction_id": "transaction_id", "txn_id": "transaction_id", "tx_id": "transaction_id",
    "transactionid": "transaction_id", "txnid": "transaction_id", "transaction": "transaction_id",
    "id": "transaction_id", "reference": "transaction_id", "ref": "transaction_id",

    "amount": "amount", "amt": "amount", "value": "amount", "transaction_amount": "amount",
    "sum": "amount", "total": "amount", "debit_credit": "amount",

    "date": "date", "txn_date": "date", "transaction_date": "date", "posted_date": "date",
    "post_date": "date", "value_date": "date", "booking_date": "date", "occurred_on": "date",

    "event_ts": "event_ts", "event_timestamp": "event_ts", "timestamp": "event_ts",
    "ts": "event_ts", "event_time": "event_ts", "updated_at": "event_ts",
    "modified_at": "event_ts", "asserted_at": "event_ts", "recorded_at": "event_ts",

    "source": "source", "src": "source", "origin_system": "source", "channel": "source",
    "provider": "source", "feed": "source",

    "action": "action", "op": "action", "operation": "action", "event_type": "action",
    "change_type": "action", "verb": "action", "type": "action",

    "event_id": "event_id", "eventid": "event_id", "evt_id": "event_id",
    "idempotency_key": "event_id", "message_id": "event_id",

    "category": "category", "cat": "category", "spend_category": "category",
    "merchant": "merchant", "merchant_name": "merchant", "payee": "merchant",
    "description": "merchant", "narrative": "merchant", "vendor": "merchant",

    "account_id": "account_id", "account": "account_id", "acct": "account_id",
    "acct_id": "account_id", "account_number": "account_id",

    "parent_transaction_id": "parent_transaction_id", "parent_id": "parent_transaction_id",
    "parent": "parent_transaction_id", "merged_into": "parent_transaction_id",
    "split_of": "parent_transaction_id",

    "currency": "currency", "ccy": "currency", "currency_code": "currency",
    "role": "role", "user_role": "role", "tier": "role", "plan": "role",
}

#: Placeholder strings that mean "absent". Exports are full of these and
#: treating ``"N/A"`` as a literal merchant name would corrupt grouping.
EMPTY_TOKENS: frozenset[str] = frozenset({
    "", "-", "--", "n/a", "na", "n.a.", "null", "none", "nil", "nan",
    "undefined", "unknown", "?", "#n/a", "(blank)", "<na>",
})

_WS_RE = re.compile(r"\s+")
_MONTHS = {
    "jan": 1, "january": 1, "feb": 2, "february": 2, "mar": 3, "march": 3,
    "apr": 4, "april": 4, "may": 5, "jun": 6, "june": 6, "jul": 7, "july": 7,
    "aug": 8, "august": 8, "sep": 9, "sept": 9, "september": 9, "oct": 10,
    "october": 10, "nov": 11, "november": 11, "dec": 12, "december": 12,
}

_ISO_DATE_RE = re.compile(r"^(\d{4})-(\d{1,2})-(\d{1,2})$")
_SLASH_ISO_RE = re.compile(r"^(\d{4})/(\d{1,2})/(\d{1,2})$")
_DOT_ISO_RE = re.compile(r"^(\d{4})\.(\d{1,2})\.(\d{1,2})$")
_COMPACT_RE = re.compile(r"^(\d{4})(\d{2})(\d{2})$")
_SLASH_AMBIG_RE = re.compile(r"^(\d{1,2})[/.\-](\d{1,2})[/.\-](\d{4})$")
_ALPHA_DMY_RE = re.compile(r"^(\d{1,2})[\s\-]([A-Za-z]{3,9})[\s\-](\d{4})$")
_ALPHA_MDY_RE = re.compile(r"^([A-Za-z]{3,9})\s+(\d{1,2}),?\s+(\d{4})$")

#: Fields the resolver can write, mapped to their normalized attribute name.
_PAYLOAD_FIELDS: tuple[str, ...] = (
    "amount_minor", "currency", "date", "category", "merchant",
    "account_id", "parent_transaction_id",
)

#: Fields whose presence makes an ``update`` meaningful. ``currency`` is absent
#: on purpose - it is only ever supplied alongside an amount, so counting it
#: would let an amount-less row masquerade as a real edit.
_ASSERTABLE_FIELDS: tuple[str, ...] = (
    "amount_minor", "date", "category", "merchant", "account_id", "parent_transaction_id",
)


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class NormalizeResult:
    """Outcome of normalizing one row.

    Collects *all* problems rather than raising on the first, so a malformed row
    can be fixed in one pass instead of one field per upload.
    """

    event: Event | None
    issues: tuple[Issue, ...]
    raw: Mapping[str, Any]
    origin: str

    @property
    def ok(self) -> bool:
        return self.event is not None

    @property
    def errors(self) -> tuple[Issue, ...]:
        return tuple(issue for issue in self.issues if issue.is_error)

    @property
    def warnings(self) -> tuple[Issue, ...]:
        return tuple(issue for issue in self.issues if not issue.is_error)

    def error_summary(self) -> str:
        return "; ".join(str(issue) for issue in self.errors) or "ok"

    def raise_if_invalid(self) -> Event:
        """Return the event, or raise :class:`ValidationError` with all issues."""
        if self.event is None:
            raise ValidationError(
                f"row could not be normalized: {self.error_summary()}",
                issues=[issue.to_dict() for issue in self.issues],
                origin=self.origin,
            )
        return self.event


# ---------------------------------------------------------------------------
# Scalar helpers
# ---------------------------------------------------------------------------

def _clean(value: Any) -> str | None:
    """Trim, collapse internal whitespace, and map placeholders to ``None``."""
    if value is None:
        return None
    if isinstance(value, str):
        text = unicodedata.normalize("NFKC", value).strip()
        text = _WS_RE.sub(" ", text)
        if text.casefold() in EMPTY_TOKENS:
            return None
        return text
    if isinstance(value, float) and value != value:      # NaN from pandas
        return None
    text = str(value).strip()
    return text or None


def remap_columns(row: Mapping[str, Any]) -> dict[str, Any]:
    """Canonicalize header names via :data:`COLUMN_ALIASES`.

    Unrecognized columns are kept under their original name so they survive
    into the audit evidence rather than being silently dropped.
    """
    remapped: dict[str, Any] = {}
    for key, value in row.items():
        if key is None:
            continue
        slug = _WS_RE.sub("_", unicodedata.normalize("NFKC", str(key)).strip().casefold())
        slug = slug.replace("-", "_").strip("_")
        canonical = COLUMN_ALIASES.get(slug)
        if canonical is None:
            remapped.setdefault(str(key), value)
            continue
        # First alias wins, so an explicit `transaction_id` beats a stray `id`.
        remapped.setdefault(canonical, value)
    return remapped


def identity_key(value: str, *, case_insensitive: bool = True) -> str:
    """Identity form of an id. ``TXN-1`` and ``txn-1`` collapse when enabled.

    Public because the state layer must resolve ``parent_transaction_id``
    references through exactly the same function that produced the keys it is
    matching against - two normalizers would eventually disagree and silently
    break dependency detection.
    """
    key = unicodedata.normalize("NFKC", value).strip()
    key = _WS_RE.sub(" ", key)
    return key.casefold() if case_insensitive else key


#: Internal alias kept for call sites within this module.
_identity_key = identity_key


def parse_date(raw: Any, *, strict_ambiguous: bool = True) -> tuple[Date, list[Issue]]:
    """Parse a business date, refusing formats that cannot be read reliably."""
    text = _clean(raw)
    if text is None:
        raise ValidationError("date is required", code="E_MISSING_DATE", field="date", value=raw)

    issues: list[Issue] = []

    # An ISO datetime in a date column: take the date part, note the truncation.
    if "T" in text or (" " in text and ":" in text):
        head = re.split(r"[T ]", text, maxsplit=1)[0]
        if _ISO_DATE_RE.match(head):
            issues.append(Issue(
                code="W_DATE_TRUNCATED",
                field="date",
                message=f"time component dropped from {text!r}",
                severity="warning",
            ))
            text = head

    for pattern in (_ISO_DATE_RE, _SLASH_ISO_RE, _DOT_ISO_RE, _COMPACT_RE):
        match = pattern.match(text)
        if match:
            if pattern is not _ISO_DATE_RE:
                issues.append(Issue(
                    code="W_DATE_ALT_FORMAT",
                    field="date",
                    message=f"{text!r} parsed as year-first; canonical form is YYYY-MM-DD",
                    severity="warning",
                ))
            return _build_date(match.group(1), match.group(2), match.group(3), raw), issues

    # Alphabetic month: unambiguous by construction.
    match = _ALPHA_DMY_RE.match(text)
    if match:
        month = _MONTHS.get(match.group(2).casefold())
        if month is None:
            raise ValidationError(
                f"unrecognized month name in {raw!r}", code="E_BAD_DATE", field="date", value=raw
            )
        issues.append(Issue(
            code="W_DATE_ALT_FORMAT", field="date",
            message=f"{text!r} parsed as DD-Mon-YYYY", severity="warning",
        ))
        return _build_date(match.group(3), month, match.group(1), raw), issues

    match = _ALPHA_MDY_RE.match(text)
    if match:
        month = _MONTHS.get(match.group(1).casefold())
        if month is None:
            raise ValidationError(
                f"unrecognized month name in {raw!r}", code="E_BAD_DATE", field="date", value=raw
            )
        issues.append(Issue(
            code="W_DATE_ALT_FORMAT", field="date",
            message=f"{text!r} parsed as Mon DD, YYYY", severity="warning",
        ))
        return _build_date(match.group(3), month, match.group(2), raw), issues

    # Numeric day/month with a 4-digit year: the dangerous case.
    match = _SLASH_AMBIG_RE.match(text)
    if match:
        first, second, year = int(match.group(1)), int(match.group(2)), match.group(3)
        if first > 12 and second <= 12:
            issues.append(Issue(
                code="W_DATE_ALT_FORMAT", field="date",
                message=f"{text!r} parsed as DD/MM/YYYY (first field > 12)",
                severity="warning",
            ))
            return _build_date(year, second, first, raw), issues
        if second > 12 and first <= 12:
            issues.append(Issue(
                code="W_DATE_ALT_FORMAT", field="date",
                message=f"{text!r} parsed as MM/DD/YYYY (second field > 12)",
                severity="warning",
            ))
            return _build_date(year, first, second, raw), issues
        if strict_ambiguous:
            raise ValidationError(
                f"date {raw!r} is ambiguous (could be DD/MM/YYYY or MM/DD/YYYY); "
                "supply YYYY-MM-DD instead",
                code="E_AMBIGUOUS_DATE", field="date", value=raw,
            )
        issues.append(Issue(
            code="W_DATE_ASSUMED_MDY", field="date",
            message=f"{text!r} is ambiguous; assumed MM/DD/YYYY "
                    "(strict_ambiguous_dates is disabled)",
            severity="warning",
        ))
        return _build_date(year, first, second, raw), issues

    raise ValidationError(
        f"date {raw!r} is not a recognized format; expected YYYY-MM-DD",
        code="E_BAD_DATE", field="date", value=raw,
    )


def _build_date(year: Any, month: Any, day: Any, raw: Any) -> Date:
    try:
        return Date(int(year), int(month), int(day))
    except ValueError as exc:
        raise ValidationError(
            f"date {raw!r} is not a real calendar date ({exc})",
            code="E_BAD_DATE", field="date", value=raw,
        ) from None


def parse_timestamp(raw: Any) -> datetime:
    """Parse an event timestamp to timezone-aware UTC.

    Accepts ISO-8601 (with ``Z`` or an offset), a space-separated variant, and
    epoch seconds/milliseconds. Naive values are assumed UTC - documented in the
    README, because reinterpreting them by local timezone would make results
    depend on the machine that ran the ingest.
    """
    text = _clean(raw)
    if text is None:
        raise ValidationError(
            "event_ts is required", code="E_MISSING_EVENT_TS", field="event_ts", value=raw
        )

    # Epoch forms. The 1e11 boundary separates seconds from milliseconds for
    # every timestamp between 1973 and 5138, which covers all real data.
    if re.fullmatch(r"-?\d{9,14}", text):
        number = int(text)
        if abs(number) >= 100_000_000_000:
            number //= 1000
        try:
            return datetime.fromtimestamp(number, tz=timezone.utc)
        except (OverflowError, OSError, ValueError):
            raise ValidationError(
                f"event_ts {raw!r} is out of range", code="E_BAD_EVENT_TS",
                field="event_ts", value=raw,
            ) from None

    candidate = text.replace(" ", "T", 1) if (" " in text and ":" in text) else text
    if candidate.endswith(("Z", "z")):
        candidate = candidate[:-1] + "+00:00"

    try:
        parsed = datetime.fromisoformat(candidate)
    except ValueError:
        # Date-only value in a timestamp column: midnight UTC is the only
        # reading, and the caller records a ts_inferred flag for it.
        try:
            parsed = datetime.combine(parse_date(text)[0], datetime.min.time())
        except ValidationError:
            raise ValidationError(
                f"event_ts {raw!r} is not a valid ISO-8601 timestamp",
                code="E_BAD_EVENT_TS", field="event_ts", value=raw,
            ) from None

    return to_utc(parsed)


def normalize_action(raw: Any) -> str:
    """Resolve an action alias to ``create`` / ``update`` / ``delete``."""
    text = _clean(raw)
    if text is None:
        raise ValidationError(
            "action is required", code="E_MISSING_ACTION", field="action", value=raw
        )
    action = ACTION_ALIASES.get(text.casefold().replace("-", "_").replace(" ", "_"))
    if action is None:
        raise ValidationError(
            f"action {raw!r} is not recognized; expected create, update or delete",
            code="E_BAD_ACTION", field="action", value=raw,
        )
    return action


def normalize_source(raw: Any) -> tuple[str, str, list[Issue]]:
    """Resolve a source alias. Returns ``(canonical, raw_text, issues)``.

    An unrecognized source is accepted at the lowest authority rather than
    rejected: losing a real transaction because a new feed appeared is worse
    than reconciling it conservatively.
    """
    text = _clean(raw)
    if text is None:
        return "unknown", "", [Issue(
            code="W_SOURCE_MISSING", field="source",
            message="source absent; treated as 'unknown' at lowest authority",
            severity="warning",
        )]
    slug = text.casefold().replace("-", "_").replace(" ", "_")
    canonical = SOURCE_ALIASES.get(slug)
    if canonical is None:
        return "unknown", text, [Issue(
            code="W_SOURCE_UNKNOWN", field="source",
            message=f"source {text!r} is not a known feed; treated as 'unknown' "
                    "at lowest authority",
            severity="warning",
        )]
    return canonical, text, []


# ---------------------------------------------------------------------------
# Row normalization
# ---------------------------------------------------------------------------

def normalize_row(
    row: Mapping[str, Any],
    *,
    config: EngineConfig = DEFAULT_CONFIG,
    origin: str = "api",
) -> NormalizeResult:
    """Normalize and validate one raw row into an :class:`Event`.

    Never raises for bad data - malformed rows come back as a result with
    ``event=None`` and a populated ``issues`` list, so a 10,000-row CSV with
    three bad lines still ingests the other 9,997.
    """
    raw = dict(row)
    mapped = remap_columns(raw)
    issues: list[Issue] = []

    def capture(fn, *args, **kwargs):
        """Run a parser, converting ValidationError into a collected Issue."""
        try:
            return fn(*args, **kwargs)
        except ValidationError as exc:
            issues.append(Issue(
                code=exc.code,
                field=str(exc.context.get("field", "")),
                message=exc.message,
            ))
            return None

    # --- identity ---------------------------------------------------------
    user_text = _clean(mapped.get("user_id"))
    if user_text is None:
        issues.append(Issue(
            code="E_MISSING_USER", field="user_id",
            message="user_id is required and cannot be blank",
        ))

    action = capture(normalize_action, mapped.get("action"))
    source, source_raw, source_issues = normalize_source(mapped.get("source"))
    issues.extend(source_issues)

    # --- payload ----------------------------------------------------------
    currency = capture(normalize_currency, mapped.get("currency"), default=config.default_currency)
    if currency is None:
        currency = config.default_currency

    supplied: set[str] = set()
    values: dict[str, Any] = {}

    amount_raw = mapped.get("amount")
    if _clean(amount_raw) is not None:
        parsed = capture(parse_amount, amount_raw, currency=currency)
        if parsed is not None:
            amount_minor, amount_warnings = parsed
            values["amount_minor"] = amount_minor
            supplied.add("amount_minor")
            supplied.add("currency")
            for warning in amount_warnings:
                code, _, message = warning.partition(":")
                issues.append(Issue(
                    code=code, field="amount", message=message, severity="warning"
                ))
    elif action == "create":
        # Required only on create. An update is a *partial* assertion: editing
        # a category must not force the caller to restate the amount, and
        # demanding it would invite exactly the copy-paste errors this engine
        # exists to reconcile. A delete needs no amount at all.
        issues.append(Issue(
            code="E_MISSING_AMOUNT", field="amount",
            message="amount is required for action='create'",
        ))

    date_value: Date | None = None
    if _clean(mapped.get("date")) is not None:
        parsed_date = capture(
            parse_date, mapped.get("date"), strict_ambiguous=config.strict_ambiguous_dates
        )
        if parsed_date is not None:
            date_value, date_issues = parsed_date
            issues.extend(date_issues)
            if not (config.min_date <= date_value <= config.max_date):
                issues.append(Issue(
                    code="E_DATE_RANGE", field="date",
                    message=f"date {date_value.isoformat()} is outside the plausible "
                            f"window {config.min_date.isoformat()}..{config.max_date.isoformat()}",
                ))
                date_value = None
            else:
                values["date"] = date_value
                supplied.add("date")
    elif action == "create":
        # A create with no date cannot be placed in any budget period.
        issues.append(Issue(
            code="E_MISSING_DATE", field="date",
            message="date is required for action='create'",
        ))

    for name in ("category", "merchant", "account_id", "parent_transaction_id"):
        text = _clean(mapped.get(name))
        if text is not None:
            values[name] = text
            supplied.add(name)

    if action == "update" and not (supplied & set(_ASSERTABLE_FIELDS)):
        # An update that asserts nothing has no meaning: there is no value to
        # reconcile, so accepting it would add a no-op row to every audit trail.
        issues.append(Issue(
            code="E_EMPTY_UPDATE", field="",
            message="action='update' must supply at least one field to change "
                    f"({', '.join(_ASSERTABLE_FIELDS)})",
        ))

    # --- timestamp --------------------------------------------------------
    flags: list[str] = []
    event_ts: datetime | None = None
    if _clean(mapped.get("event_ts")) is not None:
        event_ts = capture(parse_timestamp, mapped.get("event_ts"))
    elif date_value is not None and config.infer_missing_event_ts:
        event_ts = datetime.combine(date_value, datetime.min.time(), tzinfo=timezone.utc)
        flags.append("ts_inferred")
        issues.append(Issue(
            code="W_TS_INFERRED", field="event_ts",
            message=f"event_ts absent; inferred as {format_ts(event_ts)} from date. "
                    "Supply event_ts to order competing edits precisely.",
            severity="warning",
        ))
    else:
        issues.append(Issue(
            code="E_MISSING_EVENT_TS", field="event_ts",
            message="event_ts is required when no date is present - an event with "
                    "no clock cannot be ordered against competing edits",
        ))

    # --- transaction id ---------------------------------------------------
    txn_text = _clean(mapped.get("transaction_id"))
    if txn_text is None:
        txn_text = _synthesize_txn_id(user_text, values, action, config, issues, flags)

    # --- structural sanity ------------------------------------------------
    if txn_text and values.get("parent_transaction_id"):
        same = _identity_key(txn_text, case_insensitive=config.case_insensitive_ids) == \
               _identity_key(values["parent_transaction_id"],
                             case_insensitive=config.case_insensitive_ids)
        if same:
            issues.append(Issue(
                code="E_SELF_PARENT", field="parent_transaction_id",
                message="parent_transaction_id cannot equal transaction_id",
            ))

    if any(issue.is_error for issue in issues):
        return NormalizeResult(None, tuple(issues), raw, origin)

    assert user_text is not None and txn_text is not None
    assert action is not None and event_ts is not None

    # --- authority --------------------------------------------------------
    role = _clean(mapped.get("role"))
    role = role.casefold() if role else config.role_for(user_text)
    authority = config.priority.score(source, role)

    user_key = _identity_key(user_text, case_insensitive=config.case_insensitive_ids)
    txn_key = _identity_key(txn_text, case_insensitive=config.case_insensitive_ids)

    warnings = tuple(issue.code for issue in issues if not issue.is_error)

    event = Event(
        event_id="",                      # filled in below, content-derived
        user_id=user_text,
        user_key=user_key,
        transaction_id=txn_text,
        transaction_key=txn_key,
        action=action,
        source=source,
        event_ts=event_ts,
        currency=currency,
        amount_minor=values.get("amount_minor"),
        date=values.get("date"),
        category=values.get("category"),
        merchant=values.get("merchant"),
        account_id=values.get("account_id"),
        parent_transaction_id=values.get("parent_transaction_id"),
        supplied=frozenset(supplied),
        role=role,
        authority=authority,
        origin=origin,
        source_raw=source_raw,
        warnings=warnings,
        flags=tuple(flags),
        raw=raw,
    )

    explicit_id = _clean(mapped.get("event_id"))
    event_id = explicit_id if explicit_id else derive_event_id(event)
    if not explicit_id:
        flags.append("event_id_derived")
        event = Event(**{**_as_kwargs(event), "flags": tuple(flags)})

    return NormalizeResult(
        Event(**{**_as_kwargs(event), "event_id": event_id}), tuple(issues), raw, origin
    )


def _as_kwargs(event: Event) -> dict[str, Any]:
    """Field dict for an Event (``dataclasses.asdict`` deep-copies; we don't)."""
    return {name: getattr(event, name) for name in Event.__dataclass_fields__}


def _synthesize_txn_id(
    user_text: str | None,
    values: Mapping[str, Any],
    action: str | None,
    config: EngineConfig,
    issues: list[Issue],
    flags: list[str],
) -> str | None:
    """Derive a surrogate transaction id when the feed omitted one.

    Standard reconciliation practice: build a composite key from the fields that
    identify the transaction economically (user, date, amount, merchant). Two
    rows describing the same purchase collapse onto the same surrogate id and
    are then deduplicated by the normal conflict rules.

    ``source`` is deliberately *not* part of the key - a bank sync and a manual
    entry describing the same purchase must collapse to one transaction, which
    is the whole point of deriving a key at all.

    Refused for ``update``/``delete`` - a surrogate key derived from a *partial*
    payload would target the wrong record, and silently editing the wrong
    transaction is far worse than rejecting the row.
    """
    if not config.synthesize_missing_transaction_id:
        issues.append(Issue(
            code="E_MISSING_TXN_ID", field="transaction_id",
            message="transaction_id is required",
        ))
        return None

    if action is None:
        # The action failed to parse, so we cannot judge whether a surrogate is
        # safe. Report the plain requirement instead of speculating.
        issues.append(Issue(
            code="E_MISSING_TXN_ID", field="transaction_id",
            message="transaction_id is required",
        ))
        return None

    if action != "create":
        issues.append(Issue(
            code="E_MISSING_TXN_ID", field="transaction_id",
            message=f"transaction_id is required for action={action!r}; a surrogate id "
                    "cannot be derived from a partial payload without risking an "
                    "edit to the wrong record",
        ))
        return None

    if "amount_minor" not in values or "date" not in values:
        issues.append(Issue(
            code="E_MISSING_TXN_ID", field="transaction_id",
            message="transaction_id is absent and cannot be synthesized: amount and "
                    "date are both required to derive a surrogate key",
        ))
        return None

    surrogate = stable_id(
        user_text,
        values["date"].isoformat(),
        values["amount_minor"],
        (values.get("merchant") or "").casefold(),
        prefix="syn-",
        length=16,
    )
    flags.append("id_synthesized")
    issues.append(Issue(
        code="W_ID_SYNTHESIZED", field="transaction_id",
        message=f"transaction_id absent; derived surrogate {surrogate} from "
                "(user, date, amount, merchant)",
        severity="warning",
    ))
    return surrogate


def derive_event_id(event: Event) -> str:
    """Content-addressed event id, used when the feed supplies none.

    This is the backbone of idempotency: re-uploading the same file produces
    byte-identical ids, so the second pass is recognized as duplicate rather
    than double-counted. Derived from *semantic* content only - never from
    filename, line number or arrival order - so the same event ingested by CSV
    and by API collapses to one.
    """
    return stable_id(
        event.user_key,
        event.transaction_key,
        event.action,
        event.source,
        format_ts(event.event_ts),
        event.amount_minor,
        event.currency,
        event.date.isoformat() if event.date else None,
        event.category,
        event.merchant,
        event.account_id,
        event.parent_transaction_id,
        sorted(event.supplied),
        prefix="evt-",
        length=24,
    )


def normalize_rows(
    rows: Any,
    *,
    config: EngineConfig = DEFAULT_CONFIG,
    origin_prefix: str = "api",
) -> list[NormalizeResult]:
    """Normalize an iterable of rows, tagging each with a positional origin."""
    results: list[NormalizeResult] = []
    for index, row in enumerate(rows, start=1):
        results.append(
            normalize_row(row, config=config, origin=f"{origin_prefix}:{index}")
        )
    return results
