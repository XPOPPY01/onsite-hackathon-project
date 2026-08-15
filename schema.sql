-- Zoro conflict-resolution engine: persistent schema.
--
-- Three tables with three different lifecycles, which is why they are separate:
--
--   events        append-only, immutable. The source of truth. Replay reads this.
--   decisions     append-only. One row per resolved event; mirrors audit.log so
--                 the trail is queryable with SQL as well as greppable as JSONL.
--   transactions  derived, rewritten wholesale on save. A transaction is the
--                 *output* of folding its event log, never an accumulating row,
--                 so UPDATE-in-place would let materialized state drift from the
--                 events that produced it.
--
-- Money is INTEGER minor units everywhere. No REAL columns anywhere in this
-- schema: binary floating point cannot represent 0.10 exactly, and a cent lost
-- to representation error would make the audit trail wrong in a way no amount
-- of hashing would catch.
--
-- Timestamps and dates are ISO-8601 TEXT. SQLite has no native date type, and
-- ISO-8601 sorts lexicographically in the same order it sorts chronologically,
-- so ORDER BY on these columns is correct.

PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

-- ---------------------------------------------------------------------------
-- events: the immutable log
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS events (
    -- Arrival order. Explicitly NOT the resolution order: the fold sorts by
    -- (event_ts, action_rank, event_id), so a late event has a high seq and an
    -- early position in the timeline.
    seq                   INTEGER PRIMARY KEY,

    -- Content-addressed identity. UNIQUE is the idempotency guarantee: a
    -- re-uploaded file cannot be double-counted, because the second insert
    -- violates this constraint rather than relying on application logic.
    event_id              TEXT    NOT NULL UNIQUE,

    user_id               TEXT    NOT NULL,   -- display form, first-seen casing
    user_key              TEXT    NOT NULL,   -- identity form used for grouping
    transaction_id        TEXT    NOT NULL,
    transaction_key       TEXT    NOT NULL,

    action                TEXT    NOT NULL CHECK (action IN ('create','update','delete')),
    source                TEXT    NOT NULL,
    source_raw            TEXT    NOT NULL DEFAULT '',
    event_ts              TEXT    NOT NULL,   -- ISO-8601 UTC; when asserted

    currency              TEXT    NOT NULL,
    amount_minor          INTEGER,            -- NULL = not asserted by this event
    date                  TEXT,               -- ISO-8601 date
    category              TEXT,
    merchant              TEXT,
    account_id            TEXT,
    parent_transaction_id TEXT,

    -- JSON array of the field names this event actually asserted. Load-bearing:
    -- the fold reads nothing outside this set, which is what stops a
    -- category-only edit from nulling out an amount.
    supplied              TEXT    NOT NULL,

    role                  TEXT    NOT NULL DEFAULT 'standard',
    authority             INTEGER NOT NULL DEFAULT 0,
    origin                TEXT    NOT NULL DEFAULT 'api',

    warnings              TEXT,               -- JSON array, NULL when empty
    flags                 TEXT,               -- JSON array, NULL when empty
    raw                   TEXT                -- JSON object: the original row
);

-- The fold's working set: every event for one transaction, in timeline order.
CREATE INDEX IF NOT EXISTS ix_events_txn
    ON events (user_key, transaction_key, event_ts, action, event_id);

CREATE INDEX IF NOT EXISTS ix_events_user_ts ON events (user_key, event_ts);
CREATE INDEX IF NOT EXISTS ix_events_source  ON events (source);

-- ---------------------------------------------------------------------------
-- decisions: the queryable audit trail
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS decisions (
    seq           INTEGER PRIMARY KEY,
    event_id      TEXT    NOT NULL UNIQUE REFERENCES events (event_id),

    user_id       TEXT    NOT NULL,
    transaction_id TEXT   NOT NULL,
    action        TEXT    NOT NULL,
    source        TEXT    NOT NULL,

    -- The EVENT's timestamp, never a wall clock. Keeps the trail reproducible.
    timestamp     TEXT    NOT NULL,

    decision      TEXT    NOT NULL CHECK (
                      decision IN ('created','merged','replaced','deleted','ignored','rejected')),
    rule          TEXT    NOT NULL,
    reason        TEXT    NOT NULL,

    state_diff    TEXT,   -- JSON {"op": ..., "fields": {...}}
    evidence      TEXT,   -- JSON array of considered events
    conflicts     TEXT,   -- JSON array, NULL when clean
    anomalies     TEXT,   -- JSON array, NULL when clean
    issues        TEXT,
    warnings      TEXT,

    user_version  INTEGER NOT NULL DEFAULT 0,
    txn_version   INTEGER NOT NULL DEFAULT 0,
    state_hash    TEXT    NOT NULL DEFAULT '',

    -- Tamper-evident chain, mirroring audit.log.
    prev_hash     TEXT    NOT NULL DEFAULT '',
    record_hash   TEXT    NOT NULL DEFAULT ''
);

CREATE INDEX IF NOT EXISTS ix_decisions_user     ON decisions (user_id, timestamp);
CREATE INDEX IF NOT EXISTS ix_decisions_decision ON decisions (decision);
CREATE INDEX IF NOT EXISTS ix_decisions_rule     ON decisions (rule);
CREATE INDEX IF NOT EXISTS ix_decisions_txn      ON decisions (user_id, transaction_id);

-- ---------------------------------------------------------------------------
-- transactions: the materialized, reconciled state
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS transactions (
    user_id               TEXT    NOT NULL,
    transaction_id        TEXT    NOT NULL,
    transaction_key       TEXT    NOT NULL,

    currency              TEXT    NOT NULL,
    amount_minor          INTEGER,
    amount                TEXT,               -- exact rendering of amount_minor
    date                  TEXT,
    month                 TEXT,               -- 'YYYY-MM', precomputed for reports
    category              TEXT,
    merchant              TEXT,
    account_id            TEXT,
    parent_transaction_id TEXT,

    -- Tombstone, not a physical delete: a late create or update must reconcile
    -- against the deletion instead of silently resurrecting the row.
    deleted               INTEGER NOT NULL DEFAULT 0 CHECK (deleted IN (0,1)),
    -- Materialized from an update whose create has not arrived yet (R6).
    provisional           INTEGER NOT NULL DEFAULT 0 CHECK (provisional IN (0,1)),

    version               INTEGER NOT NULL DEFAULT 0,
    provenance            TEXT,               -- JSON: field -> who set it

    PRIMARY KEY (user_id, transaction_key)
);

CREATE INDEX IF NOT EXISTS ix_txn_live  ON transactions (user_id, deleted);
CREATE INDEX IF NOT EXISTS ix_txn_month ON transactions (user_id, month);
CREATE INDEX IF NOT EXISTS ix_txn_cat   ON transactions (user_id, category);

-- ---------------------------------------------------------------------------
-- runs: wall-clock metadata, deliberately outside the hashed audit log
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS runs (
    run_id             TEXT PRIMARY KEY,
    started_at         TEXT NOT NULL,   -- wall clock: real time, not reproducible
    finished_at        TEXT,
    engine_version     TEXT NOT NULL,
    config_fingerprint TEXT NOT NULL,   -- decisions depend on rules; pin them
    events_accepted    INTEGER NOT NULL DEFAULT 0,
    events_rejected    INTEGER NOT NULL DEFAULT 0,
    events_duplicate   INTEGER NOT NULL DEFAULT 0,
    audit_head_hash    TEXT NOT NULL DEFAULT '',
    state_hash         TEXT NOT NULL DEFAULT '',
    notes              TEXT
);
