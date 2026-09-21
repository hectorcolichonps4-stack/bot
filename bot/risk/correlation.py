"""Control de correlacion entre la senal nueva y lo que ya esta abierto.

Tres posiciones muy correlacionadas no son tres apuestas: son una sola con
triple tamano. Este modulo evita ese error contando el riesgo real.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CorrelationCheck:
    name: str
    passed: bool
    reason: str = ""
    worst_symbol: str = ""
    worst_value: float = 0.0


def returns_from_closes(closes: pd.Series, lookback: int) -> pd.Series:
    """Rendimientos logaritmicos de las ultimas ``lookback`` velas."""
    series = pd.to_numeric(closes, errors="coerce").dropna()
    if len(series) < 3:
        return pd.Series(dtype="float64")
    log_returns = np.log(series / series.shift(1)).dropna()
    return log_returns.iloc[-lookback:]


def pairwise_correlation(left: pd.Series, right: pd.Series) -> float:
    """Correlacion de Pearson alineando ambas series por longitud comun."""
    size = min(len(left), len(right))
    if size < 3:
        return 0.0
    a = np.asarray(left.iloc[-size:], dtype="float64")
    b = np.asarray(right.iloc[-size:], dtype="float64")
    if np.std(a) == 0 or np.std(b) == 0:
        return 0.0
    value = float(np.corrcoef(a, b)[0, 1])
    return 0.0 if np.isnan(value) else value


def check_correlation(
    candidate: str,
    open_symbols: Sequence[str],
    closes: Mapping[str, pd.Series],
    *,
    lookback: int,
    max_correlation: float,
) -> CorrelationCheck:
    """Rechaza la senal si correlaciona demasiado con alguna posicion abierta.

    Se usa el valor absoluto: una correlacion de -0.9 con un corto abierto es
    tan concentradora como +0.9 con un largo.
    """
    candidate = candidate.upper()
    peers = [s.upper() for s in open_symbols if s.upper() != candidate]
    if not peers:
        return CorrelationCheck("correlation", True)

    candidate_returns = returns_from_closes(closes.get(candidate, pd.Series(dtype="float64")), lookback)
    if candidate_returns.empty:
        # Sin datos suficientes no se bloquea: el limite de posiciones
        # concurrentes sigue acotando la concentracion.
        return CorrelationCheck("correlation", True, reason="sin datos suficientes para correlacionar")

    worst_symbol, worst_value = "", 0.0
    for peer in peers:
        peer_returns = returns_from_closes(closes.get(peer, pd.Series(dtype="float64")), lookback)
        if peer_returns.empty:
            continue
        value = abs(pairwise_correlation(candidate_returns, peer_returns))
        if value > worst_value:
            worst_symbol, worst_value = peer, value

    if worst_value > max_correlation:
        return CorrelationCheck(
            "correlation", False,
            f"correlacion {worst_value:.2f} con {worst_symbol} supera el maximo {max_correlation}",
            worst_symbol, worst_value,
        )
    return CorrelationCheck("correlation", True, worst_symbol=worst_symbol, worst_value=worst_value)


def correlation_matrix(closes: Mapping[str, pd.Series], lookback: int) -> pd.DataFrame:
    """Matriz de correlaciones para el dashboard y los informes."""
    series = {s: returns_from_closes(c, lookback) for s, c in closes.items()}
    series = {s: r for s, r in series.items() if not r.empty}
    symbols = sorted(series)
    matrix = pd.DataFrame(index=symbols, columns=symbols, dtype="float64")
    for row in symbols:
        for col in symbols:
            matrix.loc[row, col] = 1.0 if row == col else pairwise_correlation(series[row], series[col])
    return matrix
