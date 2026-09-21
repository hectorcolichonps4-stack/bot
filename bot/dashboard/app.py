"""Dashboard de Streamlit, estrictamente de solo lectura.

Abre la base de datos con ``mode=ro``: el dashboard no puede modificar nada
aunque alguien lo intente. Arrancarlo con:

    streamlit run bot/dashboard/app.py -- --db var/bot.sqlite
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st

from bot.config import load_config
from bot.logging.db import Database

REFRESH_SECONDS = 30


def resolve_db_path() -> str:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default=None)
    parser.add_argument("--config", default=None)
    args, _ = parser.parse_known_args()
    if args.db:
        return args.db
    try:
        return load_config(args.config).config.logging.db_path
    except Exception:  # noqa: BLE001 - sin config valida se usa la ruta por defecto
        return "var/bot.sqlite"


@st.cache_resource
def get_db(path: str) -> Database:
    return Database(path, read_only=True)


def _metric_row(db: Database) -> None:
    trades = db.trades_df()
    closed = trades.dropna(subset=["pnl"]) if not trades.empty else trades

    equity = db.equity_df()
    current_equity = float(equity["equity"].iloc[-1]) if not equity.empty else 0.0
    total_pnl = float(closed["pnl"].sum()) if not closed.empty else 0.0
    wins = int((closed["pnl"] > 0).sum()) if not closed.empty else 0
    win_rate = (wins / len(closed) * 100.0) if len(closed) else 0.0
    open_trades = int(trades["closed_at"].isna().sum()) if not trades.empty else 0

    columns = st.columns(5)
    columns[0].metric("Equity", f"{current_equity:,.2f}")
    columns[1].metric("PnL acumulado", f"{total_pnl:,.2f}")
    columns[2].metric("Operaciones cerradas", len(closed))
    columns[3].metric("Acierto", f"{win_rate:.1f}%")
    columns[4].metric("Abiertas ahora", open_trades)


def _equity_tab(db: Database) -> None:
    equity = db.equity_df()
    if equity.empty:
        st.info("Todavia no hay curva de capital registrada.")
        return
    equity["recorded_at"] = pd.to_datetime(equity["recorded_at"])
    st.line_chart(equity.set_index("recorded_at")[["equity"]])

    peak = equity["equity"].cummax()
    drawdown = (equity["equity"] - peak) / peak * 100.0
    st.caption("Drawdown (%)")
    st.area_chart(pd.DataFrame({"drawdown": drawdown.values}, index=equity["recorded_at"]))


def _trades_tab(db: Database) -> None:
    trades = db.trades_df(limit=500)
    if trades.empty:
        st.info("Sin operaciones registradas.")
        return

    closed = trades.dropna(subset=["pnl"])
    if not closed.empty:
        by_symbol = closed.groupby("symbol").agg(
            operaciones=("pnl", "size"),
            pnl=("pnl", "sum"),
            acierto=("pnl", lambda s: (s > 0).mean() * 100.0),
            r_medio=("r_multiple", "mean"),
        ).round(2)
        st.subheader("Por simbolo")
        st.dataframe(by_symbol, use_container_width=True)

        st.subheader("Motivos de salida")
        st.bar_chart(closed["exit_reason"].value_counts())

    st.subheader("Detalle")
    st.dataframe(trades, use_container_width=True, hide_index=True)


def _rejections_tab(db: Database) -> None:
    """La pestana mas util cuando el bot no opera y no se sabe por que."""
    summary = db.rejection_summary()
    if summary.empty:
        st.info("Sin rechazos registrados.")
        return

    st.subheader("Motivos agregados")
    st.dataframe(summary.head(40), use_container_width=True, hide_index=True)

    by_stage = summary.groupby("stage")["veces"].sum().sort_values(ascending=False)
    st.subheader("Por etapa")
    st.bar_chart(by_stage)

    st.subheader("Ultimos rechazos")
    st.dataframe(db.rejections_df(limit=200), use_container_width=True, hide_index=True)


def _signals_tab(db: Database) -> None:
    signals = db.signals_df(limit=300)
    if signals.empty:
        st.info("Sin senales registradas.")
        return
    st.metric("Senales ejecutadas", f"{int(signals['acted_on'].sum())} de {len(signals)}")
    st.dataframe(signals, use_container_width=True, hide_index=True)


def _health_tab(db: Database) -> None:
    runs = db.query("SELECT id, started_at, finished_at, mode, config_hash, notes FROM runs ORDER BY id DESC LIMIT 25")
    st.subheader("Ejecuciones")
    st.dataframe(runs, use_container_width=True, hide_index=True)

    st.subheader("Estado del bot")
    for key in ("last_session", "last_backtest"):
        value = db.get_state(key)
        if value is not None:
            st.write(f"**{key}**", value)

    st.subheader("Errores recientes")
    errors = db.errors_df(limit=100)
    if errors.empty:
        st.success("Sin errores registrados.")
    else:
        st.dataframe(errors[["created_at", "level", "source", "message"]], use_container_width=True, hide_index=True)


def main() -> None:
    st.set_page_config(page_title="Bot de trading", layout="wide")
    path = resolve_db_path()
    st.title("Bot de trading")
    st.caption(f"Solo lectura sobre `{path}` · actualizar con R")

    try:
        db = get_db(path)
    except FileNotFoundError:
        st.error(f"No existe la base de datos `{path}`. Ejecuta el bot al menos una vez.")
        return

    _metric_row(db)
    tabs = st.tabs(["Capital", "Operaciones", "Rechazos", "Senales", "Estado"])
    with tabs[0]:
        _equity_tab(db)
    with tabs[1]:
        _trades_tab(db)
    with tabs[2]:
        _rejections_tab(db)
    with tabs[3]:
        _signals_tab(db)
    with tabs[4]:
        _health_tab(db)


if __name__ == "__main__":
    main()
