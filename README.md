# onsite-hackathon-project
# Zoro AI — Real-Time Financial Behavior Conflict Resolution Engine

A deterministic engine that reconciles competing, out-of-order financial
events (bank syncs, third-party imports, user edits) into a single
tamper-evident transaction state. Given the same set of events, in any
arrival order, the engine always produces the same final state and the same
audit trail — that determinism guarantee is the whole point of the design.

## Status

This is the **rules/data layer**, not yet a runnable service. Seven modules
are implemented and import cleanly with zero third-party dependencies
(pure Python 3.10+ stdlib):

| Module | Role |
|---|---|
| `config.py` | Authority model, rule switches, thresholds |
| `canonical.py` | Deterministic hashing/serialization backbone |
| `errors.py` | Exception hierarchy with stable error codes |
| `money.py` | Exact integer-minor-unit money parsing |
| `models.py` | Core data structures: `Event`, `Transaction`, `Provenance`, `Decision` |
| `normalize.py` | Raw row → validated `Event` |
| `resolution.py` | `Event` log → reconciled `Transaction` state (the fold) |

**Missing:** a top-level orchestrator (`engine.py` / `store.py`) that groups
incoming events by transaction, drives `normalize → fold → decide` per event,
and persists the audit chain. `__init__.py` currently only exports
`__version__`, pending that module. See `test_engine_smoke.py` for a
hand-wired example of what that orchestration loop looks like.

## How it works, end to end

```
raw row (dict)
   │  normalize_row() / normalize_rows()
   ▼
Event                         ← immutable, validated, UTC timestamps, integer minor units
   │  group by (user_key, transaction_key)
   ▼
fold_transaction(all events for this txn)
   │  refolds the ENTIRE event log from scratch, in canonical timestamp order,
   │  every time a new event arrives — this is what makes late/out-of-order
   │  events self-correcting instead of merely detected
   ▼
FoldResult                    ← final field values + per-field Provenance + per-event effects
   │  label_decision() classifies what just happened
   ▼
DecisionLabel (decision / rule / reason)
   │  build_decision() + seal_decision()
   ▼
Decision                      ← one audit-log record, chained via SHA-256 to the previous one
```

### The two axes of conflict resolution

* **Order** — `Event.canonical_key = (event_ts, action_rank, event_id)`.
  This is the timeline events get sorted into. It has no arrival-order
  component, so replaying the same events in a different order produces the
  same timeline.
* **Authority** — `Provenance.precedence = (authority, event_ts, action_rank,
  event_id)`. This decides who *wins* a field when two events disagree.
  Authority is checked first, so a `user_edit` beats a `bank_sync` even if
  the bank event has a later timestamp or arrives after the user's.

Keeping these separate is what implements "prefer user edits over bank
syncs" correctly under out-of-order delivery — a late-arriving bank sync
still loses to an earlier user correction.

### Decision vocabulary

`created`, `merged`, `replaced`, `deleted`, `ignored` (the PRD's four core
labels plus `created`), plus `rejected` for rows that fail schema validation
before ever entering the timeline.

### Determinism guarantees

* All money is stored as **integer minor units**, never floats — floats are
  refused anywhere near a hashed payload (`canonical.py: _reject_floats`).
* `canonical_json()` sorts keys, uses compact separators, and renders
  `Decimal`/`datetime`/`date` through fixed formats — so the same logical
  value always hashes to the same bytes.
* Every `Decision` is chained: `record_hash = sha256(prev_hash + body)`.
  Editing or removing any past record invalidates every hash after it.
* Ambiguous dates (`03/04/2024`) are **rejected**, not guessed — guessing
  would be deterministic but silently wrong about half the time.
* A row with neither `date` nor `event_ts` is rejected outright — it can't
  be placed in a timeline.

### Conflict rules (R1–R11)

Implemented in `resolution.py`; identifiers are the single source of truth
(see module docstring, and the referenced `docs/RESOLUTION_RULES.md`, which
is not part of this upload):

| Rule | Meaning |
|---|---|
| R1 | Duplicate `event_id` rejected |
| R2 | Schema validation failure → `rejected` |
| R3 | First `create` for a transaction |
| R4 | Duplicate `create` on an existing transaction |
| R5 | `update` merged field-by-field by authority |
| R6 | `update` arriving before its `create` materializes a provisional record |
| R7 | `delete` applied (tombstone) |
| R8 | `delete` blocked — live children or a locked accounting period |
| R9 | `delete` arriving before its `create` records a tombstone |
| R10 | Late event annotation — doesn't change the rule, just notes reordering |
| R11 | Genuine tie (equal authority, equal timestamp) flagged as a conflict |

## Running it

No install step — everything is stdlib:

```bash
mkdir zoro_engine && cp *.py zoro_engine/
python3 -c "import zoro_engine; print(zoro_engine.__version__)"
```

Run the smoke test (see `test_engine_smoke.py`) to see the full pipeline
exercise a realistic conflict:

```bash
python3 test_engine_smoke.py
```

## What's not covered by this upload

* Persistence / storage layer (the `store` referenced by `StoreError`)
* CSV/API ingest handlers, HTTP layer
* The CLI (`zoro audit --verify`, `zoro replay`, referenced in docstrings)
* `docs/RESOLUTION_RULES.md` (referenced but not included)
* Automated tests
