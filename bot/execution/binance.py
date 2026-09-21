"""Broker real contra Binance (REST firmado).

Esta es la unica clase del proyecto que mueve dinero de verdad, y por eso es
deliberadamente estricta:

* las credenciales solo se leen de variables de entorno, nunca del YAML;
* el modo real exige una confirmacion explicita en la configuracion;
* las salidas se protegen con ordenes OCO en el exchange, de forma que el
  stop sigue vivo aunque el proceso del bot se caiga.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Mapping

import pandas as pd

from bot.config.schema import Config
from bot.data.filters import SymbolFilters
from bot.data.providers import BINANCE_MAINNET, BINANCE_TESTNET
from bot.execution.base import (
    AccountState,
    Broker,
    ExecutionError,
    Fill,
    Order,
    OrderStatus,
    OrderType,
    Position,
    Side,
)
from bot.logging.db import Database

log = logging.getLogger(__name__)

LIVE_CONFIRMATION_TOKEN = "I-UNDERSTAND-THE-RISK"


class BinanceBroker(Broker):
    """Cliente de spot de Binance con firma HMAC-SHA256."""

    name = "live"

    def __init__(
        self,
        config: Config,
        *,
        config_hash: str = "",
        db: Database | None = None,
        run_id: int | None = None,
        timeout: float = 15.0,
    ) -> None:
        exchange = config.exchange
        if not exchange.testnet and config.execution.live_confirmation != LIVE_CONFIRMATION_TOKEN:
            raise ExecutionError(
                "operativa real bloqueada: pon execution.live_confirmation "
                f"en '{LIVE_CONFIRMATION_TOKEN}' para confirmar que entiendes el riesgo"
            )

        self.api_key = os.environ.get(exchange.api_key_env, "")
        self.api_secret = os.environ.get(exchange.api_secret_env, "")
        if not self.api_key or not self.api_secret:
            raise ExecutionError(
                f"faltan credenciales: define {exchange.api_key_env} y {exchange.api_secret_env} en el entorno"
            )

        self.config = config
        self.config_hash = config_hash
        self.db = db
        self.run_id = run_id
        self.timeout = timeout
        self.base_url = (exchange.base_url or (BINANCE_TESTNET if exchange.testnet else BINANCE_MAINNET)).rstrip("/")
        self.quote_asset = exchange.quote_asset
        self.recv_window = exchange.recv_window_ms

        self._filters: dict[str, SymbolFilters] = {}
        self._positions: dict[str, Position] = {}
        self._trade_ids: dict[str, int] = {}
        self._time_offset_ms = 0
        self._sync_time()

    # -------------------------------------------------------------- transporte

    def _request(
        self,
        method: str,
        path: str,
        params: Mapping[str, Any] | None = None,
        *,
        signed: bool = False,
    ) -> Any:
        payload: dict[str, Any] = dict(params or {})
        if signed:
            payload["timestamp"] = int(time.time() * 1000) + self._time_offset_ms
            payload["recvWindow"] = self.recv_window
            query = urllib.parse.urlencode(payload)
            payload["signature"] = hmac.new(
                self.api_secret.encode("utf-8"), query.encode("utf-8"), hashlib.sha256
            ).hexdigest()

        query = urllib.parse.urlencode(payload)
        url = f"{self.base_url}{path}"
        data = None
        if method == "GET":
            if query:
                url = f"{url}?{query}"
        else:
            data = query.encode("utf-8")

        request = urllib.request.Request(url, data=data, method=method)
        request.add_header("X-MBX-APIKEY", self.api_key)
        request.add_header("User-Agent", "trading-bot/1.0")
        if data is not None:
            request.add_header("Content-Type", "application/x-www-form-urlencoded")

        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")[:400]
            raise ExecutionError(f"binance {exc.code} en {path}: {body}") from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            raise ExecutionError(f"binance inalcanzable en {path}: {exc}") from exc

    def _sync_time(self) -> None:
        """Alinea el reloj local con el del exchange.

        Una desviacion de pocos segundos basta para que Binance rechace todas
        las ordenes firmadas, y el error que devuelve no lo deja claro.
        """
        try:
            server_time = int(self._request("GET", "/api/v3/time")["serverTime"])
            self._time_offset_ms = server_time - int(time.time() * 1000)
            if abs(self._time_offset_ms) > 1000:
                log.warning("reloj desviado %d ms respecto a Binance; se compensa", self._time_offset_ms)
        except ExecutionError:
            log.warning("no se pudo sincronizar el reloj con Binance", exc_info=True)

    # ------------------------------------------------------------------ lectura

    def filters(self, symbol: str) -> SymbolFilters:
        key = symbol.upper()
        if key not in self._filters:
            payload = self._request("GET", "/api/v3/exchangeInfo", {"symbol": key})
            symbols = payload.get("symbols") or []
            if not symbols:
                raise ExecutionError(f"simbolo desconocido: {key}")
            self._filters[key] = SymbolFilters.from_binance(symbols[0])
        return self._filters[key]

    def price(self, symbol: str) -> float:
        payload = self._request("GET", "/api/v3/ticker/price", {"symbol": symbol.upper()})
        return float(payload["price"])

    def balances(self) -> dict[str, float]:
        payload = self._request("GET", "/api/v3/account", signed=True)
        return {
            entry["asset"]: float(entry["free"]) + float(entry["locked"])
            for entry in payload.get("balances", [])
            if float(entry["free"]) + float(entry["locked"]) > 0
        }

    def account(self) -> AccountState:
        balances = self.balances()
        cash = balances.get(self.quote_asset, 0.0)
        positions = tuple(self._positions.values())
        unrealized = 0.0
        for position in positions:
            try:
                unrealized += position.unrealized_pnl(self.price(position.symbol))
            except ExecutionError:
                log.warning("sin precio para %s al valorar la cuenta", position.symbol)
        equity = cash + sum(p.notional for p in positions) + unrealized
        return AccountState(cash=cash, equity=equity, positions=positions)

    def sync(self) -> None:
        """Descarta del estado local las posiciones que el exchange ya cerro.

        El bot puede reiniciarse o el OCO puede haber saltado sin nosotros: la
        verdad es el saldo del exchange, no la memoria del proceso.
        """
        self._sync_time()
        balances = self.balances()
        for symbol, position in list(self._positions.items()):
            base = self.filters(symbol).base_asset or symbol.replace(self.quote_asset, "")
            held = balances.get(base, 0.0)
            if held < position.quantity * 0.5:
                log.warning("%s ya no figura en el exchange; se cierra en el registro local", symbol)
                self._finalize_local(symbol, self.price(symbol), reason="closed_on_exchange")

    # ------------------------------------------------------------------ ordenes

    def submit(self, order: Order) -> Fill:
        filters = self.filters(order.symbol)
        quantity = filters.round_qty(order.quantity)
        price = filters.round_price(order.price) if order.price else self.price(order.symbol)
        filters.validate_order(price, quantity)

        params: dict[str, Any] = {
            "symbol": order.symbol.upper(),
            "side": "BUY" if order.side is Side.LONG else "SELL",
            "type": "MARKET" if order.order_type is OrderType.MARKET else "LIMIT",
            "quantity": _plain(quantity),
            "newClientOrderId": order.client_id,
        }
        if order.order_type is OrderType.LIMIT:
            params["price"] = _plain(price)
            params["timeInForce"] = "GTC"

        payload = self._request("POST", "/api/v3/order", params, signed=True)
        return _fill_from_payload(payload, order)

    def open_position(
        self,
        symbol: str,
        side: Side,
        quantity: float,
        *,
        stop_loss: float,
        take_profit: float,
        metadata: Mapping[str, object] | None = None,
    ) -> Position:
        symbol = symbol.upper()
        if symbol in self._positions:
            raise ExecutionError(f"ya hay una posicion abierta en {symbol}")
        if side is Side.SHORT:
            raise ExecutionError("el broker de spot no admite cortos; usa futuros o desactiva allow_short")

        fill = self.submit(Order(symbol=symbol, side=side, quantity=quantity))
        position = Position(
            symbol=symbol,
            side=side,
            quantity=fill.quantity,
            entry_price=fill.price,
            stop_loss=float(stop_loss),
            take_profit=float(take_profit),
            opened_at=fill.timestamp,
            fees_paid=fill.fee,
            config_hash=self.config_hash,
            metadata=dict(metadata or {}),
        )
        self._positions[symbol] = position

        try:
            self._place_oco(position)
        except ExecutionError:
            # Sin proteccion en el exchange la posicion queda expuesta: es
            # preferible salir de inmediato que quedarse sin stop.
            log.error("no se pudo colocar el OCO de %s; se cierra la posicion", symbol, exc_info=True)
            self.close_position(symbol, reason="oco_failed")
            raise

        if self.db is not None:
            self._trade_ids[symbol] = self.db.open_trade(
                symbol=symbol,
                side=side.value,
                quantity=position.quantity,
                entry_price=position.entry_price,
                stop_loss=position.stop_loss,
                take_profit=position.take_profit,
                opened_at=position.opened_at,
                mode=self.name,
                config_hash=self.config_hash,
                fees=position.fees_paid,
                signal_id=(metadata or {}).get("signal_id"),
                run_id=self.run_id,
            )
        return position

    def _place_oco(self, position: Position) -> None:
        """Deja stop y objetivo vivos en el exchange (OCO)."""
        filters = self.filters(position.symbol)
        stop = filters.round_price(position.stop_loss)
        # El precio limite del stop se separa un poco para que llegue a cruzar.
        stop_limit = filters.round_price(position.stop_loss * 0.999)
        target = filters.round_price(position.take_profit)

        self._request(
            "POST",
            "/api/v3/order/oco",
            {
                "symbol": position.symbol,
                "side": "SELL",
                "quantity": _plain(position.quantity),
                "price": _plain(target),
                "stopPrice": _plain(stop),
                "stopLimitPrice": _plain(stop_limit),
                "stopLimitTimeInForce": "GTC",
            },
            signed=True,
        )

    def close_position(self, symbol: str, *, reason: str = "manual") -> Fill | None:
        symbol = symbol.upper()
        position = self._positions.get(symbol)
        if position is None:
            return None

        self.cancel_open_orders(symbol)
        fill = self.submit(
            Order(symbol=symbol, side=position.side.opposite, quantity=position.quantity, reduce_only=True)
        )
        self._finalize_local(symbol, fill.price, reason=reason, fee=fill.fee)
        return fill

    def cancel_open_orders(self, symbol: str) -> None:
        """Retira el OCO antes de cerrar a mercado, para no dejar ordenes huerfanas."""
        try:
            self._request("DELETE", "/api/v3/openOrders", {"symbol": symbol.upper()}, signed=True)
        except ExecutionError as exc:
            # -2011 significa que no habia nada que cancelar.
            if "-2011" not in str(exc):
                raise

    def _finalize_local(self, symbol: str, exit_price: float, *, reason: str, fee: float = 0.0) -> None:
        position = self._positions.pop(symbol, None)
        if position is None:
            return
        gross = (exit_price - position.entry_price) * position.quantity * position.side.sign
        net = gross - fee - position.fees_paid
        risk = position.risk_per_unit * position.quantity

        trade_id = self._trade_ids.pop(symbol, None)
        if self.db is not None and trade_id is not None:
            self.db.close_trade(
                trade_id,
                exit_price=exit_price,
                closed_at=pd.Timestamp.now(tz="UTC"),
                pnl=net,
                fees=fee + position.fees_paid,
                r_multiple=net / risk if risk > 0 else 0.0,
                exit_reason=reason,
            )
        log.info("CERRADA (live) %s por %s @ %.6f pnl=%.2f", symbol, reason, exit_price, net)


def _plain(value: float) -> str:
    """Formatea sin notacion cientifica: Binance rechaza ``1e-05``."""
    return f"{value:.8f}".rstrip("0").rstrip(".") or "0"


def _fill_from_payload(payload: Mapping[str, Any], order: Order) -> Fill:
    fills = payload.get("fills") or []
    executed = float(payload.get("executedQty") or order.quantity)
    if fills:
        notional = sum(float(f["price"]) * float(f["qty"]) for f in fills)
        quantity = sum(float(f["qty"]) for f in fills)
        price = notional / quantity if quantity else 0.0
        fee = sum(float(f.get("commission", 0.0)) for f in fills)
    else:
        quantity = executed
        cummulative = float(payload.get("cummulativeQuoteQty") or 0.0)
        price = cummulative / quantity if quantity else float(payload.get("price") or 0.0)
        fee = 0.0

    status_map = {
        "FILLED": OrderStatus.FILLED,
        "PARTIALLY_FILLED": OrderStatus.PARTIALLY_FILLED,
        "NEW": OrderStatus.NEW,
        "CANCELED": OrderStatus.CANCELED,
        "REJECTED": OrderStatus.REJECTED,
    }
    return Fill(
        order_id=str(payload.get("orderId", order.client_id)),
        symbol=order.symbol.upper(),
        side=order.side,
        quantity=quantity,
        price=price,
        fee=fee,
        timestamp=pd.Timestamp(int(payload.get("transactTime", time.time() * 1000)), unit="ms", tz="UTC"),
        status=status_map.get(str(payload.get("status", "FILLED")), OrderStatus.FILLED),
        raw=dict(payload),
    )
