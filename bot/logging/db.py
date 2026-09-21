"""Persistencia en SQLite: trades, senales, rechazos, errores y estado.

La base de datos es la memoria del bot y la unica fuente del dashboard. Se
abre en modo WAL para que el dashboard pueda leer mientras el bot escribe.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import traceback as tb_module
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import pandas as pd

SCHEMA_PATH = Path(__file__).with_name("schema.sql")
SCHEMA_VERSION = 1


def _iso(value: Any) -> str | None:
    if value is None:
        return None
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC").isoformat()


def _json(value: Any) -> str:
    return json.dumps(value, default=str, ensure_ascii=False, sort_keys=True)


class Database:
    """Acceso a SQLite con esquema autoaplicado y escrituras serializadas."""

    def __init__(self, path: str | Path, *, read_only: bool = False) -> None:
        self.path = Path(path)
        self.read_only = read_only
        self._lock = threading.Lock()

        if read_only:
            if not self.path.exists():
                raise FileNotFoundError(f"no existe la base de datos: {self.path}")
            uri = f"file:{self.path}?mode=ro"
            self._conn = sqlite3.connect(uri, uri=True, check_same_thread=False)
        else:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(self.path, check_same_thread=False, isolation_level=None)
            self._migrate()

        self._conn.row_factory = sqlite3.Row

    # ------------------------------------------------------------ infraestructura

    def _migrate(self) -> None:
        self._conn.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
        current = self._conn.execute("SELECT MAX(version) FROM schema_version").fetchone()[0]
        if current is None:
            self._conn.execute(
                "INSERT INTO schema_version(version, applied_at) VALUES (?, ?)",
                (SCHEMA_VERSION, _iso(pd.Timestamp.now(tz="UTC"))),
            )

    @contextmanager
    def _write(self) -> Iterator[sqlite3.Connection]:
        if self.read_only:
            raise RuntimeError("la base de datos esta abierta en solo lectura")
        with self._lock:
            yield self._conn

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "Database":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -------------------------------------------------------------------- runs

    def start_run(self, *, mode: str, config_hash: str, config: Mapping[str, Any], notes: str = "") -> int:
        with self._write() as conn:
            cursor = conn.execute(
                "INSERT INTO runs(started_at, mode, config_hash, config_json, notes) VALUES (?,?,?,?,?)",
                (_iso(pd.Timestamp.now(tz="UTC")), mode, config_hash, _json(config), notes),
            )
            return int(cursor.lastrowid)

    def finish_run(self, run_id: int, notes: str = "") -> None:
        with self._write() as conn:
            conn.execute(
                "UPDATE runs SET finished_at = ?, notes = COALESCE(NULLIF(?, ''), notes) WHERE id = ?",
                (_iso(pd.Timestamp.now(tz="UTC")), notes, run_id),
            )

    # ----------------------------------------------------------------- senales

    def log_signal(self, signal: Any, *, run_id: int | None = None, acted_on: bool = False) -> int:
        with self._write() as conn:
            cursor = conn.execute(
                """INSERT INTO signals(run_id, created_at, bar_time, symbol, side, entry_price,
                                       stop_loss, take_profit, score, components, snapshot,
                                       acted_on, config_hash)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    run_id,
                    _iso(pd.Timestamp.now(tz="UTC")),
                    _iso(signal.timestamp),
                    signal.symbol,
                    signal.side.value,
                    float(signal.entry_price),
                    float(signal.stop_loss),
                    float(signal.take_profit),
                    float(signal.score),
                    _json(dict(signal.components)),
                    _json(dict(signal.snapshot)),
                    int(acted_on),
                    signal.config_hash,
                ),
            )
            return int(cursor.lastrowid)

    def mark_signal_acted(self, signal_id: int) -> None:
        with self._write() as conn:
            conn.execute("UPDATE signals SET acted_on = 1 WHERE id = ?", (signal_id,))

    # ---------------------------------------------------------------- rechazos

    def log_rejection(
        self,
        *,
        symbol: str,
        bar_time: Any,
        stage: str,
        reason: str,
        detail: Mapping[str, Any] | None = None,
        config_hash: str = "",
        run_id: int | None = None,
    ) -> int:
        """Registra por que NO se opero. Es el log mas util para diagnosticar."""
        with self._write() as conn:
            cursor = conn.execute(
                """INSERT INTO rejections(run_id, created_at, bar_time, symbol, stage, reason, detail, config_hash)
                   VALUES (?,?,?,?,?,?,?,?)""",
                (
                    run_id,
                    _iso(pd.Timestamp.now(tz="UTC")),
                    _iso(bar_time),
                    symbol.upper(),
                    stage,
                    reason,
                    _json(dict(detail or {})),
                    config_hash,
                ),
            )
            return int(cursor.lastrowid)

    # ------------------------------------------------------------------ trades

    def open_trade(
        self,
        *,
        symbol: str,
        side: str,
        quantity: float,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        opened_at: Any,
        mode: str,
        config_hash: str,
        fees: float = 0.0,
        signal_id: int | None = None,
        run_id: int | None = None,
    ) -> int:
        with self._write() as conn:
            cursor = conn.execute(
                """INSERT INTO trades(run_id, signal_id, symbol, side, quantity, entry_price,
                                      stop_loss, take_profit, opened_at, fees, mode, config_hash)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    run_id, signal_id, symbol.upper(), side, float(quantity), float(entry_price),
                    float(stop_loss), float(take_profit), _iso(opened_at), float(fees), mode, config_hash,
                ),
            )
            return int(cursor.lastrowid)

    def close_trade(
        self,
        trade_id: int,
        *,
        exit_price: float,
        closed_at: Any,
        pnl: float,
        fees: float,
        r_multiple: float,
        exit_reason: str,
    ) -> None:
        with self._write() as conn:
            conn.execute(
                """UPDATE trades
                      SET exit_price = ?, closed_at = ?, pnl = ?, fees = ?, r_multiple = ?, exit_reason = ?
                    WHERE id = ?""",
                (float(exit_price), _iso(closed_at), float(pnl), float(fees), float(r_multiple), exit_reason, trade_id),
            )

    # ------------------------------------------------------------------ errores

    def log_error(
        self,
        *,
        source: str,
        message: str,
        level: str = "ERROR",
        exc: BaseException | None = None,
        config_hash: str = "",
        run_id: int | None = None,
    ) -> int:
        traceback_text = "".join(tb_module.format_exception(type(exc), exc, exc.__traceback__)) if exc else ""
        with self._write() as conn:
            cursor = conn.execute(
                "INSERT INTO errors(run_id, created_at, level, source, message, traceback, config_hash) VALUES (?,?,?,?,?,?,?)",
                (run_id, _iso(pd.Timestamp.now(tz="UTC")), level, source, message, traceback_text, config_hash),
            )
            return int(cursor.lastrowid)

    # ------------------------------------------------------------------- estado

    def set_state(self, key: str, value: Any) -> None:
        with self._write() as conn:
            conn.execute(
                """INSERT INTO bot_state(key, value, updated_at) VALUES (?,?,?)
                   ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at""",
                (key, _json(value), _iso(pd.Timestamp.now(tz="UTC"))),
            )

    def get_state(self, key: str, default: Any = None) -> Any:
        row = self._conn.execute("SELECT value FROM bot_state WHERE key = ?", (key,)).fetchone()
        return json.loads(row["value"]) if row else default

    def record_equity(
        self,
        *,
        equity: float,
        cash: float,
        exposure: float = 0.0,
        open_positions: int = 0,
        recorded_at: Any = None,
        run_id: int | None = None,
    ) -> None:
        with self._write() as conn:
            conn.execute(
                "INSERT INTO equity_curve(run_id, recorded_at, equity, cash, exposure, open_positions) VALUES (?,?,?,?,?,?)",
                (run_id, _iso(recorded_at or pd.Timestamp.now(tz="UTC")), float(equity), float(cash), float(exposure), int(open_positions)),
            )

    # ------------------------------------------------------------------ lectura

    def query(self, sql: str, params: Sequence[Any] = ()) -> pd.DataFrame:
        """Consulta libre devuelta como DataFrame (la usa el dashboard)."""
        return pd.read_sql_query(sql, self._conn, params=tuple(params))

    def trades_df(self, *, config_hash: str | None = None, limit: int | None = None) -> pd.DataFrame:
        sql = "SELECT * FROM trades"
        params: list[Any] = []
        if config_hash:
            sql += " WHERE config_hash = ?"
            params.append(config_hash)
        sql += " ORDER BY opened_at DESC"
        if limit:
            sql += f" LIMIT {int(limit)}"
        return self.query(sql, params)

    def rejections_df(self, *, limit: int = 500) -> pd.DataFrame:
        return self.query("SELECT * FROM rejections ORDER BY created_at DESC LIMIT ?", (limit,))

    def signals_df(self, *, limit: int = 500) -> pd.DataFrame:
        return self.query("SELECT * FROM signals ORDER BY created_at DESC LIMIT ?", (limit,))

    def errors_df(self, *, limit: int = 200) -> pd.DataFrame:
        return self.query("SELECT * FROM errors ORDER BY created_at DESC LIMIT ?", (limit,))

    def equity_df(self, *, run_id: int | None = None) -> pd.DataFrame:
        if run_id is None:
            return self.query("SELECT * FROM equity_curve ORDER BY recorded_at")
        return self.query("SELECT * FROM equity_curve WHERE run_id = ? ORDER BY recorded_at", (run_id,))

    def rejection_summary(self) -> pd.DataFrame:
        """Motivos de rechazo agregados: responde a "por que no opera el bot"."""
        return self.query(
            """SELECT stage, reason, COUNT(*) AS veces, MAX(created_at) AS ultima
                 FROM rejections GROUP BY stage, reason ORDER BY veces DESC"""
        )
