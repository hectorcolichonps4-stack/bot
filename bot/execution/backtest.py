"""Broker de backtest: la cuenta avanza al ritmo de las velas historicas."""

from __future__ import annotations

from typing import Mapping

import pandas as pd

from bot.execution.simulated import SimulatedBroker


class BacktestBroker(SimulatedBroker):
    """Ejecuta contra datos historicos, con el reloj controlado por el motor.

    El precio de referencia es el cierre de la vela en curso. Las entradas se
    envian en la vela siguiente a la senal (lo decide el motor), para no operar
    a un precio que en tiempo real todavia no se conocia.
    """

    name = "backtest"

    def advance(self, timestamp: pd.Timestamp, bars: Mapping[str, Mapping[str, float]]) -> None:
        """Mueve el reloj y actualiza los precios con la vela recibida."""
        self.now = pd.Timestamp(timestamp)
        for symbol, bar in bars.items():
            self.set_price(symbol, float(bar["close"]))

    def sync(self) -> None:
        """No hay nada que sincronizar: el estado simulado ya es la verdad."""
        return None
