"""Estrategias de referencia contra las que medir el bot.

Comprar y mantener dice si la estrategia aporta algo sobre no hacer nada. Los
cruces de EMA y el filtro RSI/MACD dicen algo mas incomodo: si los gates y la
puntuacion aportan algo sobre las dos reglas mas simples que usan los mismos
indicadores. Todas pagan las mismas comisiones y el mismo deslizamiento.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from bot.backtesting.metrics import Metrics, compute_metrics
from bot.execution.base import ClosedTrade, Side
from bot.indicators import ema, macd, rsi

# Devuelve, para cada vela, si la referencia quiere estar dentro (True) o fuera.
Rule = Callable[[pd.DataFrame], pd.Series]


@dataclass(frozen=True)
class BaselineResult:
    """Curva de capital y metricas de una referencia."""

    name: str
    equity_curve: pd.Series
    trades: list[ClosedTrade]
    metrics: Metrics

    @property
    def trade_count(self) -> int:
        return len(self.trades)


# ------------------------------------------------------------------- reglas

def rule_buy_and_hold(frame: pd.DataFrame) -> pd.Series:
    return pd.Series(True, index=frame.index)


def make_ema_rule(fast: int = 20, slow: int = 50) -> Rule:
    """Dentro mientras la EMA rapida este por encima de la lenta."""

    def rule(frame: pd.DataFrame) -> pd.Series:
        quick = ema(frame["close"], fast)
        slowly = ema(frame["close"], slow)
        return (quick > slowly).where(quick.notna() & slowly.notna(), False)

    return rule


def make_rsi_macd_rule(rsi_period: int = 14, rsi_threshold: float = 50.0,
                       fast: int = 12, slow: int = 26, signal: int = 9) -> Rule:
    """Dentro con RSI por encima del umbral y el histograma MACD positivo."""

    def rule(frame: pd.DataFrame) -> pd.Series:
        strength = rsi(frame["close"], rsi_period)
        histogram = macd(frame["close"], fast, slow, signal)["hist"]
        inside = (strength > rsi_threshold) & (histogram > 0)
        return inside.where(strength.notna() & histogram.notna(), False)

    return rule


# ------------------------------------------------------------------- motor

def run_rule(
    frame: pd.DataFrame,
    rule: Rule,
    *,
    symbol: str,
    initial_balance: float,
    fee_bps: float = 7.5,
    slippage_bps: float = 4.0,
) -> tuple[pd.Series, list[ClosedTrade]]:
    """Aplica una regla de dentro/fuera a un simbolo y devuelve su capital.

    La decision de la vela N se ejecuta en la apertura de la N+1, igual que en
    el motor principal: comparar una referencia sin ese retardo contra una
    estrategia que si lo tiene seria hacer trampa a favor de la referencia.
    """
    if frame.empty:
        return pd.Series(dtype="float64"), []

    data = frame.reset_index(drop=True)
    inside = rule(data).fillna(False).astype(bool).shift(1).fillna(False)

    fee_rate = fee_bps / 10_000.0
    slip_rate = slippage_bps / 10_000.0
    open_price = data["open"].to_numpy(dtype="float64")
    close_price = data["close"].to_numpy(dtype="float64")
    signal_in = inside.to_numpy(dtype=bool)

    cash = float(initial_balance)
    units = 0.0
    entry_price = 0.0
    entry_time = None
    entry_fee = 0.0
    trades: list[ClosedTrade] = []
    equity = np.empty(len(data), dtype="float64")

    for i in range(len(data)):
        want_in, holding = bool(signal_in[i]), units > 0

        if want_in and not holding:
            price = open_price[i] * (1.0 + slip_rate)
            units = cash / (price * (1.0 + fee_rate))
            entry_fee = units * price * fee_rate
            cash -= units * price + entry_fee
            entry_price, entry_time = price, data["open_time"].iloc[i]
        elif not want_in and holding:
            price = open_price[i] * (1.0 - slip_rate)
            exit_fee = units * price * fee_rate
            cash += units * price - exit_fee
            trades.append(
                ClosedTrade(
                    symbol=symbol, side=Side.LONG, quantity=units,
                    entry_price=entry_price, exit_price=price,
                    opened_at=entry_time, closed_at=data["open_time"].iloc[i],
                    pnl=(price - entry_price) * units - exit_fee - entry_fee,
                    fees=exit_fee + entry_fee, exit_reason="rule_exit",
                )
            )
            units = 0.0

        equity[i] = cash + units * close_price[i]

    if units > 0:
        price = close_price[-1] * (1.0 - slip_rate)
        exit_fee = units * price * fee_rate
        trades.append(
            ClosedTrade(
                symbol=symbol, side=Side.LONG, quantity=units,
                entry_price=entry_price, exit_price=price,
                opened_at=entry_time, closed_at=data["open_time"].iloc[-1],
                pnl=(price - entry_price) * units - exit_fee - entry_fee,
                fees=exit_fee + entry_fee, exit_reason="session_end",
            )
        )
        equity[-1] = cash + units * price

    return pd.Series(equity, index=pd.to_datetime(data["open_time"])), trades


def run_baseline(
    frames: Mapping[str, pd.DataFrame],
    rule: Rule,
    *,
    name: str,
    initial_balance: float,
    fee_bps: float = 7.5,
    slippage_bps: float = 4.0,
) -> BaselineResult:
    """Cartera equiponderada: el capital se reparte a partes iguales al inicio.

    No hay rebalanceo posterior, que es la version honesta de "compra lo mismo
    de cada cosa y dejalo estar"; rebalancear anadiria decisiones que ninguna
    de estas referencias toma.
    """
    usable = {symbol: frame for symbol, frame in frames.items() if not frame.empty}
    if not usable:
        return BaselineResult(name, pd.Series(dtype="float64"), [], Metrics())

    slice_balance = initial_balance / len(usable)
    curves: list[pd.Series] = []
    trades: list[ClosedTrade] = []

    for symbol, frame in usable.items():
        curve, symbol_trades = run_rule(
            frame, rule, symbol=symbol, initial_balance=slice_balance,
            fee_bps=fee_bps, slippage_bps=slippage_bps,
        )
        if not curve.empty:
            curves.append(curve)
            trades.extend(symbol_trades)

    if not curves:
        return BaselineResult(name, pd.Series(dtype="float64"), [], Metrics())

    # Union de indices y relleno hacia delante: un simbolo que aun no cotiza
    # aporta su parte intacta, no un hueco que rompa la suma.
    combined = pd.concat(curves, axis=1).sort_index()
    combined = combined.ffill().fillna(slice_balance)
    equity = combined.sum(axis=1)

    trades.sort(key=lambda t: t.closed_at)
    return BaselineResult(name, equity, trades, compute_metrics(equity, trades))


def standard_baselines(
    frames: Mapping[str, pd.DataFrame],
    *,
    initial_balance: float,
    fee_bps: float = 7.5,
    slippage_bps: float = 4.0,
    ema_fast: int = 20,
    ema_slow: int = 50,
) -> list[BaselineResult]:
    """Las tres referencias que se comparan siempre con la estrategia."""
    definitions: Sequence[tuple[str, Rule]] = (
        ("Comprar y mantener", rule_buy_and_hold),
        (f"Cruce EMA {ema_fast}/{ema_slow}", make_ema_rule(ema_fast, ema_slow)),
        ("RSI>50 + MACD>0", make_rsi_macd_rule()),
    )
    return [
        run_baseline(
            frames, rule, name=name, initial_balance=initial_balance,
            fee_bps=fee_bps, slippage_bps=slippage_bps,
        )
        for name, rule in definitions
    ]
