"""Filtros del exchange: tick size, step size, minimos notionales.

Redondear mal un precio o una cantidad es la causa numero uno de ordenes
rechazadas en produccion, asi que la normalizacion vive en un solo sitio.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import ROUND_DOWN, ROUND_HALF_UP, Decimal
from typing import Any, Mapping


class FilterError(ValueError):
    """La orden no cumple los filtros del simbolo."""


@dataclass(frozen=True)
class SymbolFilters:
    """Restricciones de un simbolo tal y como las publica el exchange."""

    symbol: str
    tick_size: Decimal = Decimal("0.01")
    step_size: Decimal = Decimal("0.00001")
    min_qty: Decimal = Decimal("0")
    max_qty: Decimal = Decimal("1000000000")
    min_notional: Decimal = Decimal("10")
    base_asset: str = ""
    quote_asset: str = "USDT"

    @classmethod
    def permissive(cls, symbol: str) -> "SymbolFilters":
        """Filtros laxos para backtests y tests, donde no hay exchange real."""
        return cls(
            symbol=symbol.upper(),
            tick_size=Decimal("0.00000001"),
            step_size=Decimal("0.00000001"),
            min_qty=Decimal("0"),
            min_notional=Decimal("0"),
        )

    @classmethod
    def from_binance(cls, payload: Mapping[str, Any]) -> "SymbolFilters":
        """Construye los filtros a partir de una entrada de ``exchangeInfo``."""
        by_type = {f["filterType"]: f for f in payload.get("filters", [])}
        price = by_type.get("PRICE_FILTER", {})
        lot = by_type.get("LOT_SIZE", {})
        notional = by_type.get("NOTIONAL") or by_type.get("MIN_NOTIONAL") or {}
        return cls(
            symbol=str(payload["symbol"]).upper(),
            tick_size=Decimal(str(price.get("tickSize", "0.01"))),
            step_size=Decimal(str(lot.get("stepSize", "0.00001"))),
            min_qty=Decimal(str(lot.get("minQty", "0"))),
            max_qty=Decimal(str(lot.get("maxQty", "1000000000"))),
            min_notional=Decimal(str(notional.get("minNotional", notional.get("notional", "10")))),
            base_asset=str(payload.get("baseAsset", "")),
            quote_asset=str(payload.get("quoteAsset", "USDT")),
        )

    def round_price(self, price: float | Decimal) -> float:
        """Ajusta el precio al tick mas cercano (half-up)."""
        return float(_quantize(Decimal(str(price)), self.tick_size, ROUND_HALF_UP))

    def round_qty(self, qty: float | Decimal) -> float:
        """Ajusta la cantidad al step, siempre hacia abajo.

        Redondear hacia arriba puede dejar la orden por encima del riesgo
        autorizado, asi que aqui se trunca deliberadamente.
        """
        return float(_quantize(Decimal(str(qty)), self.step_size, ROUND_DOWN))

    def validate_order(self, price: float, qty: float) -> None:
        """Lanza ``FilterError`` si la orden no seria aceptada."""
        qty_dec = Decimal(str(qty))
        price_dec = Decimal(str(price))
        if qty_dec <= 0:
            raise FilterError(f"{self.symbol}: cantidad {qty} no positiva")
        if qty_dec < self.min_qty:
            raise FilterError(f"{self.symbol}: cantidad {qty} por debajo de minQty {self.min_qty}")
        if qty_dec > self.max_qty:
            raise FilterError(f"{self.symbol}: cantidad {qty} por encima de maxQty {self.max_qty}")
        notional = price_dec * qty_dec
        if notional < self.min_notional:
            raise FilterError(
                f"{self.symbol}: notional {notional} por debajo del minimo {self.min_notional}"
            )

    def conform(self, price: float, qty: float) -> tuple[float, float]:
        """Redondea y valida en un solo paso; devuelve (precio, cantidad)."""
        rounded_price = self.round_price(price)
        rounded_qty = self.round_qty(qty)
        self.validate_order(rounded_price, rounded_qty)
        return rounded_price, rounded_qty


def _quantize(value: Decimal, step: Decimal, rounding: str) -> Decimal:
    if step <= 0:
        return value
    return (value / step).quantize(Decimal("1"), rounding=rounding) * step
