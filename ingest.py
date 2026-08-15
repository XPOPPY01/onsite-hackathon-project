"""Reading raw events from CSV, JSON and JSONL.

Purely mechanical: this layer turns bytes into a list of dicts and records where
each one came from. It makes no judgements about content - validation is
:mod:`zoro_engine.normalize`'s job - because a reader that also validates tends
to abort a 10,000-row file over one bad cell.

Every row carries an ``origin`` like ``csv:march.csv:47`` that follows it all the
way into ``audit.log``, so any decision can be traced back to the exact line
that caused it.
"""

from __future__ import annotations

import csv
import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .config import DEFAULT_MAX_UPLOAD_BYTES
from .errors import IngestError

#: Key under which CSV columns beyond the header row are collected, rather than
#: silently dropped.
RESTKEY = "__extra_columns__"

#: Delimiters worth sniffing. Financial exports use all four.
_DELIMITERS = ",;\t|"

#: Cap on sniffing work for very wide files.
_SNIFF_BYTES = 64 * 1024


@dataclass(frozen=True, slots=True)
class RawRow:
    """One unparsed input record and its provenance."""

    data: Mapping[str, Any]
    origin: str


def _decode(payload: bytes | str, *, origin: str) -> str:
    """Decode to text, tolerating a UTF-8 BOM and falling back to latin-1.

    A hard decode failure on one stray byte would reject an entire upload, so
    the fallback keeps the file readable and lets normalization judge the
    contents.
    """
    if isinstance(payload, str):
        return payload.lstrip("﻿")
    for encoding in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            return payload.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise IngestError(f"could not decode {origin} as text", origin=origin)


def _check_size(payload: bytes | str, *, origin: str, max_bytes: int) -> None:
    size = len(payload if isinstance(payload, bytes) else payload.encode("utf-8"))
    if size > max_bytes:
        raise IngestError(
            f"{origin} is {size} bytes, over the {max_bytes} byte limit",
            code="E_TOO_LARGE",
            origin=origin,
            size=size,
            limit=max_bytes,
        )


def sniff_delimiter(sample: str) -> str:
    """Best-guess CSV delimiter.

    :class:`csv.Sniffer` is tried first; on failure we count candidates in the
    header line, which is more reliable than the default comma for the
    semicolon-delimited exports common outside the US.
    """
    try:
        return csv.Sniffer().sniff(sample, delimiters=_DELIMITERS).delimiter
    except csv.Error:
        header = sample.splitlines()[0] if sample.splitlines() else ""
        counts = {candidate: header.count(candidate) for candidate in _DELIMITERS}
        best = max(counts, key=lambda key: counts[key])
        return best if counts[best] > 0 else ","


def rows_from_csv(
    payload: bytes | str,
    *,
    origin: str = "csv",
    max_bytes: int = DEFAULT_MAX_UPLOAD_BYTES,
) -> list[RawRow]:
    """Parse CSV into rows, one :class:`RawRow` per data line.

    Fully blank lines are skipped (trailing newlines are not errors). Columns
    beyond the header are preserved under :data:`RESTKEY` rather than dropped,
    so nothing in the source file disappears without trace.
    """
    _check_size(payload, origin=origin, max_bytes=max_bytes)
    text = _decode(payload, origin=origin)
    if not text.strip():
        return []

    delimiter = sniff_delimiter(text[:_SNIFF_BYTES])
    reader = csv.DictReader(
        io.StringIO(text, newline=""), delimiter=delimiter, restkey=RESTKEY
    )
    if reader.fieldnames is None:
        return []

    rows: list[RawRow] = []
    for row in reader:
        if all(value in (None, "") for value in row.values()):
            continue
        # DictReader.line_num is the physical line, which is what a user needs
        # in order to find the row in their spreadsheet.
        rows.append(RawRow(data=dict(row), origin=f"{origin}:{reader.line_num}"))
    return rows


def rows_from_json(
    payload: bytes | str | Mapping[str, Any] | Sequence[Any],
    *,
    origin: str = "json",
    max_bytes: int = DEFAULT_MAX_UPLOAD_BYTES,
) -> list[RawRow]:
    """Parse a JSON object, a JSON array, or an already-decoded structure.

    A bare object is accepted as a single event, which is the shape the PRD
    specifies for ``POST /events``.
    """
    if isinstance(payload, (bytes, str)):
        _check_size(payload, origin=origin, max_bytes=max_bytes)
        text = _decode(payload, origin=origin)
        try:
            decoded: Any = json.loads(text)
        except json.JSONDecodeError as exc:
            raise IngestError(
                f"{origin} is not valid JSON: {exc.msg} (line {exc.lineno}, column {exc.colno})",
                code="E_BAD_JSON",
                origin=origin,
            ) from None
    else:
        decoded = payload

    # A wrapper object such as {"events": [...]} is the natural request body.
    if isinstance(decoded, Mapping):
        for key in ("events", "rows", "data", "records"):
            inner = decoded.get(key)
            if isinstance(inner, Sequence) and not isinstance(inner, (str, bytes)):
                decoded = inner
                break

    if isinstance(decoded, Mapping):
        return [RawRow(data=dict(decoded), origin=f"{origin}:1")]

    if isinstance(decoded, Sequence) and not isinstance(decoded, (str, bytes)):
        rows: list[RawRow] = []
        for index, item in enumerate(decoded, start=1):
            if not isinstance(item, Mapping):
                raise IngestError(
                    f"{origin}: element {index} is {type(item).__name__}, expected an object",
                    code="E_BAD_JSON",
                    origin=f"{origin}:{index}",
                )
            rows.append(RawRow(data=dict(item), origin=f"{origin}:{index}"))
        return rows

    raise IngestError(
        f"{origin}: expected a JSON object or array, got {type(decoded).__name__}",
        code="E_BAD_JSON",
        origin=origin,
    )


def rows_from_jsonl(
    payload: bytes | str,
    *,
    origin: str = "jsonl",
    max_bytes: int = DEFAULT_MAX_UPLOAD_BYTES,
) -> list[RawRow]:
    """Parse newline-delimited JSON, one event per line."""
    _check_size(payload, origin=origin, max_bytes=max_bytes)
    text = _decode(payload, origin=origin)

    rows: list[RawRow] = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            item = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise IngestError(
                f"{origin}:{lineno} is not valid JSON: {exc.msg}",
                code="E_BAD_JSON",
                origin=f"{origin}:{lineno}",
            ) from None
        if not isinstance(item, Mapping):
            raise IngestError(
                f"{origin}:{lineno} is {type(item).__name__}, expected an object",
                code="E_BAD_JSON",
                origin=f"{origin}:{lineno}",
            )
        rows.append(RawRow(data=dict(item), origin=f"{origin}:{lineno}"))
    return rows


def rows_from_path(
    path: str | Path,
    *,
    max_bytes: int = DEFAULT_MAX_UPLOAD_BYTES,
) -> list[RawRow]:
    """Read a file, choosing the reader by extension then by content."""
    target = Path(path)
    if not target.exists():
        raise IngestError(f"{target} does not exist", code="E_NOT_FOUND", path=str(target))
    if not target.is_file():
        raise IngestError(f"{target} is not a file", code="E_NOT_FILE", path=str(target))

    payload = target.read_bytes()
    origin = f"{target.suffix.lstrip('.') or 'file'}:{target.name}"
    suffix = target.suffix.casefold()

    if suffix in {".json"}:
        return rows_from_json(payload, origin=f"json:{target.name}", max_bytes=max_bytes)
    if suffix in {".jsonl", ".ndjson"}:
        return rows_from_jsonl(payload, origin=f"jsonl:{target.name}", max_bytes=max_bytes)
    if suffix in {".csv", ".tsv", ".txt", ""}:
        return rows_from_csv(payload, origin=f"csv:{target.name}", max_bytes=max_bytes)

    # Unknown extension: decide on the first non-space character rather than
    # refusing a perfectly good file over its name.
    head = _decode(payload[:_SNIFF_BYTES], origin=origin).lstrip()
    if head.startswith(("{", "[")):
        return rows_from_json(payload, origin=f"json:{target.name}", max_bytes=max_bytes)
    return rows_from_csv(payload, origin=f"csv:{target.name}", max_bytes=max_bytes)


def rows_from_paths(
    paths: Iterable[str | Path],
    *,
    max_bytes: int = DEFAULT_MAX_UPLOAD_BYTES,
) -> list[RawRow]:
    """Read several files in the order given.

    Order is preserved deliberately: it is the *arrival* order the engine will
    process, and reproducing a run means replaying the same sequence.
    """
    collected: list[RawRow] = []
    for path in paths:
        collected.extend(rows_from_path(path, max_bytes=max_bytes))
    return collected
