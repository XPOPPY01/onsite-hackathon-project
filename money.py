"""Money handling: exact integer minor units, never floats.

The PRD types ``amount`` as ``float``, and floats are how real feeds arrive, so
the parser accepts them at the boundary. Internally the engine stores
**integer minor units** (cents for USD) because:

* ``0.1 + 0.2 != 0.3`` in binary floating point, so float sums are
  order-dependent - which would break the determinism requirement outright;
* equality comparison drives conflict detection, and ``19.99 != 19.99`` can be
  true for floats that took different paths to the same value;
* audit records must be byte-stable, and ``repr(float)`` is not a stable
  contract across operations.

Conversion happens exactly once, at ingest, via :func:`parse_amount`.
"""

from __future__ import annotations

import math
import re
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from typing import Any

from .errors import ValidationError

#: Minor-unit exponent per ISO-4217 currency. Anything unlisted defaults to 2.
MINOR_DIGITS: dict[str, int] = {
    "USD": 2, "EUR": 2, "GBP": 2, "INR": 2, "CAD": 2, "AUD": 2,
    "CHF": 2, "SGD": 2, "AED": 2, "CNY": 2, "BRL": 2, "ZAR": 2,
    "JPY": 0, "KRW": 0, "VND": 0, "CLP": 0,   # zero-decimal currencies
    "BHD": 3, "KWD": 3, "JOD": 3, "TND": 3,   # three-decimal currencies
}

DEFAULT_CURRENCY = "USD"

#: Hard sanity bound (in major units). Beyond this we assume a corrupt feed
#: rather than a real transaction, and reject the row instead of poisoning
#: downstream aggregates.
MAX_ABS_MAJOR = Decimal("1000000000000")  # 1 trillion

_CURRENCY_RE = re.compile(r"^[A-Z]{3}$")
_STRIP_RE = re.compile(r"[\s,_  ]")           # spaces, thousands seps
_SYMBOL_RE = re.compile(r"^[\$£€¥₹]|[\$£€¥₹]$")
_ACCOUNTING_NEG_RE = re.compile(r"^\((.*)\)$")           # (123.45) -> -123.45
_TRAILING_NEG_RE = re.compile(r"^(.*?)-$")               # 123.45-  -> -123.45


def minor_digits(currency: str) -> int:
    """Minor-unit exponent for ``currency`` (defaults to 2)."""
    return MINOR_DIGITS.get(currency.upper(), 2)


def normalize_currency(raw: Any, *, default: str = DEFAULT_CURRENCY) -> str:
    """Validate and upper-case a currency code."""
    if raw is None:
        return default
    text = str(raw).strip().upper()
    if not text:
        return default
    if not _CURRENCY_RE.match(text):
        raise ValidationError(
            f"currency must be a 3-letter ISO-4217 code, got {raw!r}",
            code="E_BAD_CURRENCY",
            field="currency",
            value=raw,
        )
    return text


def parse_amount(raw: Any, *, currency: str = DEFAULT_CURRENCY) -> tuple[int, list[str]]:
    """Parse a raw amount into integer minor units.

    Accepts the messy shapes that real financial exports contain: thousands
    separators, currency symbols, accounting-style parenthesised negatives,
    trailing minus signs, and floats.

    Returns ``(minor_units, warnings)``. A warning is emitted when the value
    carried more precision than the currency supports and had to be rounded -
    the value is still accepted, but the audit trail records that we touched it.

    Raises :class:`ValidationError` for anything that is not a finite number.
    """
    warnings: list[str] = []

    if raw is None or (isinstance(raw, str) and not raw.strip()):
        raise ValidationError(
            "amount is required", code="E_MISSING_AMOUNT", field="amount", value=raw
        )

    if isinstance(raw, bool):
        # bool is an int subclass; treating True as 1.00 is never intended.
        raise ValidationError(
            f"amount must be numeric, got boolean {raw!r}",
            code="E_BAD_AMOUNT",
            field="amount",
            value=raw,
        )

    if isinstance(raw, float):
        if not math.isfinite(raw):
            raise ValidationError(
                f"amount must be finite, got {raw!r}",
                code="E_BAD_AMOUNT",
                field="amount",
                value=str(raw),
            )
        # str() round-trips a float to its shortest exact repr, which is a far
        # better Decimal seed than Decimal(float) binary expansion.
        candidate = Decimal(str(raw))
    elif isinstance(raw, int):
        candidate = Decimal(raw)
    elif isinstance(raw, Decimal):
        if not raw.is_finite():
            raise ValidationError(
                f"amount must be finite, got {raw!r}",
                code="E_BAD_AMOUNT",
                field="amount",
                value=str(raw),
            )
        candidate = raw
    else:
        candidate = _parse_amount_text(str(raw), raw)

    if not candidate.is_finite():
        raise ValidationError(
            f"amount must be finite, got {raw!r}",
            code="E_BAD_AMOUNT",
            field="amount",
            value=str(raw),
        )

    if abs(candidate) > MAX_ABS_MAJOR:
        raise ValidationError(
            f"amount {candidate} exceeds the sanity bound of {MAX_ABS_MAJOR}",
            code="E_AMOUNT_RANGE",
            field="amount",
            value=str(raw),
        )

    digits = minor_digits(currency)
    scale = Decimal(10) ** digits
    scaled = candidate * scale

    if scaled != scaled.to_integral_value():
        # More precision than the currency can express. ROUND_HALF_UP is the
        # convention for retail financial amounts (and is deterministic).
        rounded = scaled.quantize(Decimal(1), rounding=ROUND_HALF_UP)
        warnings.append(
            f"W_ROUNDED:amount {candidate} rounded to {rounded / scale} "
            f"({digits}dp for {currency})"
        )
        scaled = rounded

    return int(scaled), warnings


def _parse_amount_text(text: str, original: Any) -> Decimal:
    """Parse the string forms of an amount."""
    cleaned = text.strip()

    negative = False
    match = _ACCOUNTING_NEG_RE.match(cleaned)
    if match:                       # (123.45) accounting negative
        negative = True
        cleaned = match.group(1).strip()

    cleaned = _SYMBOL_RE.sub("", cleaned).strip()
    cleaned = _STRIP_RE.sub("", cleaned)

    match = _TRAILING_NEG_RE.match(cleaned)
    if match:                       # 123.45- mainframe-style negative
        negative = True
        cleaned = match.group(1)

    if cleaned.startswith("+"):
        cleaned = cleaned[1:]

    if not cleaned or cleaned in {"-", "."}:
        raise ValidationError(
            f"amount could not be parsed from {original!r}",
            code="E_BAD_AMOUNT",
            field="amount",
            value=str(original),
        )

    try:
        value = Decimal(cleaned)
    except InvalidOperation:
        raise ValidationError(
            f"amount could not be parsed from {original!r}",
            code="E_BAD_AMOUNT",
            field="amount",
            value=str(original),
        ) from None

    if not value.is_finite():
        # Decimal happily parses "NaN" and "Infinity"; money cannot be either.
        raise ValidationError(
            f"amount must be finite, got {original!r}",
            code="E_BAD_AMOUNT",
            field="amount",
            value=str(original),
        )

    return -value if negative else value


def format_amount(minor: int | None, currency: str = DEFAULT_CURRENCY) -> str | None:
    """Render minor units as a fixed-precision decimal string.

    A *string*, not a float: this value goes into JSON output, the audit log and
    hashed payloads, so it must be exact. ``-1234`` USD -> ``"-12.34"``.
    """
    if minor is None:
        return None
    digits = minor_digits(currency)
    if digits == 0:
        return str(minor)
    scale = 10 ** digits
    sign = "-" if minor < 0 else ""
    whole, frac = divmod(abs(minor), scale)
    return f"{sign}{whole}.{frac:0{digits}d}"


def to_decimal(minor: int | None, currency: str = DEFAULT_CURRENCY) -> Decimal | None:
    """Convert minor units back to an exact :class:`Decimal` in major units."""
    if minor is None:
        return None
    return Decimal(minor) / (Decimal(10) ** minor_digits(currency))
