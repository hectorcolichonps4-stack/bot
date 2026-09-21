-- Esquema de la base de datos operativa del bot.
-- Toda fila relevante guarda config_hash: sin el, un resultado historico no
-- se puede reatribuir a los parametros que lo produjeron.

PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS schema_version (
    version     INTEGER PRIMARY KEY,
    applied_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS runs (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at   TEXT NOT NULL,
    finished_at  TEXT,
    mode         TEXT NOT NULL,
    config_hash  TEXT NOT NULL,
    config_json  TEXT NOT NULL,
    notes        TEXT DEFAULT ''
);

CREATE TABLE IF NOT EXISTS signals (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id        INTEGER REFERENCES runs(id),
    created_at    TEXT NOT NULL,
    bar_time      TEXT NOT NULL,
    symbol        TEXT NOT NULL,
    side          TEXT NOT NULL,
    entry_price   REAL NOT NULL,
    stop_loss     REAL NOT NULL,
    take_profit   REAL NOT NULL,
    score         REAL NOT NULL,
    components    TEXT NOT NULL,
    snapshot      TEXT NOT NULL,
    acted_on      INTEGER NOT NULL DEFAULT 0,
    config_hash   TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS rejections (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id       INTEGER REFERENCES runs(id),
    created_at   TEXT NOT NULL,
    bar_time     TEXT NOT NULL,
    symbol       TEXT NOT NULL,
    stage        TEXT NOT NULL,   -- data | gate | score | levels | riesgo
    reason       TEXT NOT NULL,
    detail       TEXT NOT NULL DEFAULT '{}',
    config_hash  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS trades (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id        INTEGER REFERENCES runs(id),
    signal_id     INTEGER REFERENCES signals(id),
    symbol        TEXT NOT NULL,
    side          TEXT NOT NULL,
    quantity      REAL NOT NULL,
    entry_price   REAL NOT NULL,
    exit_price    REAL,
    stop_loss     REAL NOT NULL,
    take_profit   REAL NOT NULL,
    opened_at     TEXT NOT NULL,
    closed_at     TEXT,
    pnl           REAL,
    fees          REAL NOT NULL DEFAULT 0,
    r_multiple    REAL,
    exit_reason   TEXT,
    mode          TEXT NOT NULL,
    config_hash   TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS errors (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id       INTEGER REFERENCES runs(id),
    created_at   TEXT NOT NULL,
    level        TEXT NOT NULL,
    source       TEXT NOT NULL,
    message      TEXT NOT NULL,
    traceback    TEXT DEFAULT '',
    config_hash  TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS bot_state (
    key         TEXT PRIMARY KEY,
    value       TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS equity_curve (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id      INTEGER REFERENCES runs(id),
    recorded_at TEXT NOT NULL,
    equity      REAL NOT NULL,
    cash        REAL NOT NULL,
    exposure    REAL NOT NULL DEFAULT 0,
    open_positions INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_signals_symbol_time  ON signals(symbol, bar_time);
CREATE INDEX IF NOT EXISTS idx_rejections_stage     ON rejections(stage, created_at);
CREATE INDEX IF NOT EXISTS idx_rejections_symbol    ON rejections(symbol, created_at);
CREATE INDEX IF NOT EXISTS idx_trades_symbol        ON trades(symbol, opened_at);
CREATE INDEX IF NOT EXISTS idx_trades_config        ON trades(config_hash);
CREATE INDEX IF NOT EXISTS idx_trades_closed        ON trades(closed_at);
CREATE INDEX IF NOT EXISTS idx_equity_run           ON equity_curve(run_id, recorded_at);
CREATE INDEX IF NOT EXISTS idx_errors_time          ON errors(created_at);
