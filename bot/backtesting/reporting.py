"""Informes derivados: actividad mensual, cortes por periodo y comparativas."""

from __future__ import annotations

from typing import Mapping, Sequence

import pandas as pd

from bot.backtesting.metrics import Metrics, max_drawdown
from bot.execution.base import ClosedTrade


def trades_per_month(trades: Sequence[ClosedTrade]) -> pd.DataFrame:
    """Operaciones cerradas por mes, con su resultado y su tasa de acierto.

    Un bot que concentra toda su actividad en dos meses no es el mismo que uno
    que opera de forma sostenida, aunque el retorno total coincida.
    """
    if not trades:
        return pd.DataFrame(columns=["mes", "operaciones", "pnl", "acierto_%", "R_medio"])

    frame = pd.DataFrame(
        [{"closed_at": t.closed_at, "pnl": t.pnl, "win": t.is_win, "r": t.r_multiple} for t in trades]
    )
    # El bot vive en UTC; se quita la zona antes de agrupar porque los
    # periodos de pandas no la conservan y avisarian en cada llamada.
    closed = pd.to_datetime(frame["closed_at"], utc=True).dt.tz_localize(None)
    frame["mes"] = closed.dt.to_period("M").astype(str)
    grouped = frame.groupby("mes").agg(
        operaciones=("pnl", "size"),
        pnl=("pnl", "sum"),
        acierto_pct=("win", lambda s: s.mean() * 100.0),
        R_medio=("r", "mean"),
    ).reset_index()
    grouped = grouped.rename(columns={"acierto_pct": "acierto_%"})
    return grouped.round({"pnl": 2, "acierto_%": 1, "R_medio": 2})


def monthly_activity_summary(trades: Sequence[ClosedTrade]) -> dict[str, float]:
    """Resumen de la regularidad: media, mediana y meses sin operar."""
    monthly = trades_per_month(trades)
    if monthly.empty:
        return {"meses": 0, "media": 0.0, "mediana": 0.0, "maximo": 0, "meses_sin_operar": 0}

    periods = pd.PeriodIndex(monthly["mes"], freq="M")
    span = pd.period_range(periods.min(), periods.max(), freq="M")
    return {
        "meses": len(span),
        "media": round(float(monthly["operaciones"].sum()) / max(len(span), 1), 1),
        "mediana": float(monthly["operaciones"].median()),
        "maximo": int(monthly["operaciones"].max()),
        "meses_sin_operar": int(len(span) - len(monthly)),
    }


def period_label(moment: pd.Timestamp, freq: str) -> str:
    """Etiqueta legible del periodo al que pertenece una marca temporal."""
    stamp = pd.Timestamp(moment)
    if freq == "semester":
        return f"{stamp.year}-H{1 if stamp.month <= 6 else 2}"
    if freq == "quarter":
        return f"{stamp.year}-Q{(stamp.month - 1) // 3 + 1}"
    if freq == "year":
        return str(stamp.year)
    if freq == "month":
        return f"{stamp.year}-{stamp.month:02d}"
    raise ValueError(f"periodo no soportado: {freq}")


def split_by_period(
    equity: pd.Series,
    trades: Sequence[ClosedTrade],
    *,
    freq: str = "semester",
) -> pd.DataFrame:
    """Trocea la curva por periodos naturales y mide dentro de cada uno.

    Los semestres son naturales (enero-junio y julio-diciembre), no ventanas
    deslizantes desde la primera vela: solo asi son comparables entre anos.

    Un retorno total bueno puede ocultar un unico semestre extraordinario y
    varios mediocres; verlos por separado es lo que distingue una estrategia
    de una racha.
    """
    columnas = ["periodo", "retorno_%", "maxDD_%", "operaciones", "acierto_%", "pnl"]
    if equity.empty:
        return pd.DataFrame(columns=columnas)

    curve = equity.copy()
    curve.index = pd.to_datetime(curve.index)
    etiquetas = pd.Index([period_label(t, freq) for t in curve.index], name="periodo")

    por_periodo: dict[str, list[ClosedTrade]] = {}
    for trade in trades:
        por_periodo.setdefault(period_label(trade.closed_at, freq), []).append(trade)

    rows: list[dict[str, object]] = []
    for label in etiquetas.unique():
        chunk = curve[etiquetas == label]
        if len(chunk) < 2:
            continue
        window = por_periodo.get(label, [])
        wins = [t for t in window if t.is_win]
        rows.append(
            {
                "periodo": label,
                "retorno_%": round((float(chunk.iloc[-1]) / float(chunk.iloc[0]) - 1.0) * 100.0, 2),
                "maxDD_%": round(max_drawdown(chunk), 2),
                "operaciones": len(window),
                "acierto_%": round(len(wins) / len(window) * 100.0, 1) if window else 0.0,
                "pnl": round(sum(t.pnl for t in window), 2),
            }
        )
    return pd.DataFrame(rows, columns=columnas)


def comparison_table(entries: Mapping[str, tuple[Metrics, int]]) -> pd.DataFrame:
    """Tabla comparativa entre la estrategia y sus referencias."""
    rows = [
        {
            "estrategia": name,
            "retorno_%": round(metrics.total_return_pct, 2),
            "CAGR_%": round(metrics.cagr_pct, 2),
            "maxDD_%": round(metrics.max_drawdown_pct, 2),
            "Sharpe": round(metrics.sharpe, 2),
            "Calmar": round(metrics.calmar, 2),
            "operaciones": trade_count,
            "acierto_%": round(metrics.win_rate_pct, 1),
            "PF": round(metrics.profit_factor, 2) if metrics.profit_factor != float("inf") else float("inf"),
            "comisiones": round(metrics.total_fees, 2),
        }
        for name, (metrics, trade_count) in entries.items()
    ]
    return pd.DataFrame(rows)


def equity_to_frame(curves: Mapping[str, pd.Series]) -> pd.DataFrame:
    """Alinea varias curvas de capital en un unico DataFrame comparable."""
    usable = {name: curve for name, curve in curves.items() if not curve.empty}
    if not usable:
        return pd.DataFrame()
    frame = pd.concat(usable, axis=1).sort_index()
    return frame.ffill()
