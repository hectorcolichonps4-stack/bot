"""Dimensionado de posicion.

Regla base: arriesgar un porcentaje fijo del capital por operacion, donde el
riesgo lo define la distancia al stop. El tamano nunca se deriva del precio de
entrada ni del capital disponible sin mas, porque eso hace que la perdida de
una operacion dependa de la volatilidad del par.
"""

from __future__ import annotations

from dataclasses import dataclass

from bot.data.filters import FilterError, SymbolFilters


@dataclass(frozen=True)
class SizingResult:
    """Cantidad propuesta junto con los limites que la recortaron."""

    quantity: float
    risk_amount: float
    notional: float
    capped_by: str | None = None
    reason: str = ""

    @property
    def viable(self) -> bool:
        return self.quantity > 0


def position_size(
    *,
    equity: float,
    entry_price: float,
    stop_loss: float,
    risk_per_trade_pct: float,
    max_position_pct: float,
    available_cash: float,
    filters: SymbolFilters,
) -> SizingResult:
    """Calcula la cantidad a operar respetando riesgo, exposicion y filtros.

    El orden de los recortes importa: primero el riesgo (cuanto puedo perder),
    luego el tamano maximo de posicion, luego el efectivo real y por ultimo el
    redondeo del exchange, que siempre trunca hacia abajo.
    """
    if equity <= 0:
        return SizingResult(0.0, 0.0, 0.0, "equity", "equity no positivo")

    risk_per_unit = abs(entry_price - stop_loss)
    if risk_per_unit <= 0:
        return SizingResult(0.0, 0.0, 0.0, "stop", "el stop coincide con la entrada")

    risk_amount = equity * risk_per_trade_pct / 100.0
    quantity = risk_amount / risk_per_unit
    capped_by: str | None = None

    max_notional = equity * max_position_pct / 100.0
    if quantity * entry_price > max_notional:
        quantity = max_notional / entry_price
        capped_by = "max_position_pct"

    if quantity * entry_price > available_cash:
        quantity = available_cash / entry_price
        capped_by = "available_cash"

    quantity = filters.round_qty(quantity)
    if quantity <= 0:
        return SizingResult(0.0, risk_amount, 0.0, "step_size", "la cantidad se anula al redondear al step del exchange")

    try:
        filters.validate_order(entry_price, quantity)
    except FilterError as exc:
        return SizingResult(0.0, risk_amount, quantity * entry_price, "exchange_filters", str(exc))

    return SizingResult(
        quantity=quantity,
        risk_amount=quantity * risk_per_unit,
        notional=quantity * entry_price,
        capped_by=capped_by,
        reason="" if capped_by is None else f"tamano recortado por {capped_by}",
    )
