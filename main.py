#!/usr/bin/env python3
"""Orquestador del bot: backtest | paper | live.

    python main.py backtest --symbols BTCUSDT,ETHUSDT
    python main.py paper
    python main.py live            # exige confirmacion explicita en el YAML

Este fichero solo ensambla piezas: elige el proveedor de datos y el broker
segun el modo, y delega en /backtesting o en /runner. Ninguna regla de
negocio vive aqui.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bot.backtesting import (
    BacktestEngine,
    WalkForwardAnalysis,
    compare,
    run_monte_carlo,
)
from bot.config import ConfigError, LoadedConfig, load_config
from bot.data import (
    BinanceRestProvider,
    DataFeed,
    MarketDataProvider,
    SyntheticProvider,
    timeframe_to_timedelta,
)
from bot.execution import LIVE_CONFIRMATION_TOKEN, BinanceBroker, PaperBroker
from bot.logging import Database, setup_logging
from bot.runner import TradingSession

EXIT_OK, EXIT_CONFIG, EXIT_RUNTIME, EXIT_ABORTED = 0, 2, 3, 130


# --------------------------------------------------------------------- CLI

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="Bot de trading: backtest, paper y live sobre la misma estrategia.",
    )
    parser.add_argument("mode", choices=("backtest", "paper", "live"), help="modo de ejecucion")
    parser.add_argument("-c", "--config", default=None, help="ruta del settings.yaml")
    parser.add_argument("--symbols", default=None, help="lista separada por comas que sustituye a data.symbols")
    parser.add_argument("--timeframe", default=None, help="temporalidad de las velas (p. ej. 15m)")
    parser.add_argument("--balance", type=float, default=None, help="capital inicial para backtest y paper")
    parser.add_argument("--start", default=None, help="inicio del backtest (YYYY-MM-DD)")
    parser.add_argument("--end", default=None, help="fin del backtest (YYYY-MM-DD)")
    parser.add_argument("--offline", action="store_true", help="usa velas sinteticas en vez de la red")
    parser.add_argument("--no-cache-refresh", action="store_true", help="no descarga: usa solo la cache local")
    parser.add_argument("--iterations", type=int, default=None, help="numero maximo de ciclos en paper/live")
    parser.add_argument("--walk-forward", action="store_true", help="fuerza el analisis walk-forward")
    parser.add_argument("--monte-carlo", action="store_true", help="fuerza la simulacion de Monte Carlo")
    parser.add_argument("--db", default=None, help="ruta de la base de datos SQLite")
    return parser


def overrides_from_args(args: argparse.Namespace) -> dict[str, Any]:
    """Traduce los argumentos de linea de comandos a un parche de configuracion."""
    patch: dict[str, Any] = {"execution": {"mode": args.mode}}
    data: dict[str, Any] = {}
    if args.symbols:
        data["symbols"] = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    if args.timeframe:
        data["timeframe"] = args.timeframe
    if data:
        patch["data"] = data

    backtesting: dict[str, Any] = {}
    if args.balance is not None:
        backtesting["initial_balance"] = args.balance
    if args.start:
        backtesting["start"] = args.start
    if args.end:
        backtesting["end"] = args.end
    if args.walk_forward:
        backtesting["walk_forward"] = {"enabled": True}
    if args.monte_carlo:
        backtesting["monte_carlo"] = {"enabled": True}
    if backtesting:
        patch["backtesting"] = backtesting

    if args.db:
        patch["logging"] = {"db_path": args.db}
    return patch


def build_provider(loaded: LoadedConfig, *, offline: bool, mode: str) -> MarketDataProvider:
    """Elige la fuente de velas: sinteticas en offline, Binance en el resto."""
    if not offline:
        exchange = loaded.config.exchange
        return BinanceRestProvider(testnet=exchange.testnet, base_url=exchange.base_url)

    # En backtest las velas sinteticas se anclan al final del rango pedido y
    # se generan tantas como haga falta para cubrirlo (con un tope, porque el
    # objetivo es probar el circuito completo, no simular anos de 1m).
    backtesting = loaded.config.backtesting
    anchor = None
    bars = 4000
    if mode == "backtest" and backtesting.end:
        anchor = pd.Timestamp(backtesting.end, tz="UTC")
        if backtesting.start:
            step = timeframe_to_timedelta(loaded.config.data.timeframe)
            span = pd.Timestamp(backtesting.end) - pd.Timestamp(backtesting.start)
            needed = int(span / step) + loaded.config.data.warmup_bars
            bars = max(1000, min(needed, 30_000))
    return SyntheticProvider(bars=bars, seed=7, anchor=anchor)


# ----------------------------------------------------------------- modos

def run_backtest(loaded: LoadedConfig, feed: DataFeed, db: Database, run_id: int, args: argparse.Namespace) -> int:
    config = loaded.config.backtesting
    start = pd.Timestamp(config.start) if config.start else None
    end = pd.Timestamp(config.end) if config.end else None

    frames = feed.bulk_history(start=start, end=end, refresh=not args.no_cache_refresh)
    print("\nvelas cargadas: " + ", ".join(f"{s}={len(f)}" for s, f in frames.items()))

    result = BacktestEngine(loaded, db=db, run_id=run_id).run(
        frames, initial_balance=config.initial_balance, start=start, end=end
    )

    _print_header("RESULTADO DEL BACKTEST")
    print(result.metrics.summary())
    print(f"comisiones pagadas: {result.metrics.total_fees:,.2f} | hash de configuracion: {result.config_hash}")

    counts = result.rejection_counts()
    if not counts.empty:
        _print_header("POR QUE NO SE OPERO (top 10)")
        print(counts.head(10).to_string())

    if not result.trades_frame.empty:
        _print_header("ULTIMAS OPERACIONES")
        print(result.trades_frame.tail(10).to_string(index=False))

    benchmark_symbol = config.benchmark_symbol
    if benchmark_symbol in frames:
        comparison = compare(
            result.equity_curve,
            result.metrics,
            frames[benchmark_symbol],
            symbol=benchmark_symbol,
            initial_balance=config.initial_balance,
            fee_bps=loaded.config.execution.fee_bps,
        )
        _print_header("BENCHMARK")
        print(comparison.summary())

    if config.monte_carlo.enabled and result.trades:
        simulation = run_monte_carlo(
            result.trades,
            initial_balance=config.initial_balance,
            iterations=config.monte_carlo.iterations,
            seed=config.monte_carlo.seed,
        )
        _print_header("MONTE CARLO")
        print(simulation.summary())

    if config.walk_forward.enabled:
        _print_header("WALK-FORWARD")
        report = WalkForwardAnalysis(
            loaded,
            # Grid pequeno a proposito: cada combinacion es un backtest
            # completo por ventana, y cuanto mas grande es el grid mas facil
            # resulta encontrar parametros que solo funcionan en el pasado.
            grid={
                "scoring.min_score": [55.0, 60.0, 65.0],
                "risk.stop_atr_multiple": [1.5, 1.8, 2.2],
            },
        ).run(frames)
        print(report.summary())
        frame = report.to_frame()
        if not frame.empty:
            print()
            print(frame.to_string(index=False))

    db.set_state("last_backtest", {"config_hash": result.config_hash, **result.metrics.as_dict()})
    return EXIT_OK


def run_paper(loaded: LoadedConfig, feed: DataFeed, db: Database, run_id: int, args: argparse.Namespace) -> int:
    broker = PaperBroker(
        feed,
        initial_balance=loaded.config.backtesting.initial_balance,
        fee_bps=loaded.config.execution.fee_bps,
        slippage_bps=loaded.config.execution.slippage_bps,
        config_hash=loaded.config_hash,
        db=db,
        run_id=run_id,
    )
    session = TradingSession(loaded, feed=feed, broker=broker, db=db, run_id=run_id)
    session.install_signal_handlers()
    session.run_forever(max_iterations=args.iterations)

    _print_header("RESUMEN DE LA SESION (PAPER)")
    account = broker.account()
    print(f"equity {account.equity:,.2f} | efectivo {account.cash:,.2f} | posiciones abiertas {len(account.positions)}")
    print(f"operaciones cerradas: {len(broker.closed_trades)}")
    return EXIT_OK


def run_live(loaded: LoadedConfig, feed: DataFeed, db: Database, run_id: int, args: argparse.Namespace) -> int:
    exchange = loaded.config.exchange
    if not exchange.testnet and loaded.config.execution.live_confirmation != LIVE_CONFIRMATION_TOKEN:
        print(
            "\nOPERATIVA REAL BLOQUEADA.\n"
            f"Pon execution.live_confirmation: {LIVE_CONFIRMATION_TOKEN} en el settings.yaml\n"
            "para confirmar que entiendes que este modo mueve dinero real.\n",
            file=sys.stderr,
        )
        return EXIT_CONFIG

    destino = "TESTNET" if exchange.testnet else "DINERO REAL"
    print(f"\n*** MODO LIVE contra {destino} | config {loaded.config_hash} ***\n")

    broker = BinanceBroker(loaded.config, config_hash=loaded.config_hash, db=db, run_id=run_id)
    session = TradingSession(loaded, feed=feed, broker=broker, db=db, run_id=run_id)
    session.install_signal_handlers()
    session.run_forever(max_iterations=args.iterations)
    return EXIT_OK


# ------------------------------------------------------------------ main

def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        loaded = load_config(args.config, overrides=overrides_from_args(args))
    except ConfigError as exc:
        print(f"\nconfiguracion invalida:\n{exc}\n", file=sys.stderr)
        return EXIT_CONFIG

    db = Database(loaded.config.logging.db_path)
    run_id = db.start_run(
        mode=args.mode,
        config_hash=loaded.config_hash,
        config=loaded.config.model_dump(mode="json"),
        notes=" ".join(argv or sys.argv[1:]),
    )
    log = setup_logging(loaded.config.logging, db=db, config_hash=loaded.config_hash, run_id=run_id)

    provider = build_provider(loaded, offline=args.offline, mode=args.mode)
    feed = DataFeed(provider, loaded.config)
    handlers = {"backtest": run_backtest, "paper": run_paper, "live": run_live}

    try:
        return handlers[args.mode](loaded, feed, db, run_id, args)
    except KeyboardInterrupt:
        log.warning("interrumpido por el usuario")
        return EXIT_ABORTED
    except Exception as exc:  # noqa: BLE001 - se registra y se sale con codigo propio
        log.error("fallo no recuperable en modo %s: %s", args.mode, exc, exc_info=True)
        return EXIT_RUNTIME
    finally:
        db.finish_run(run_id)
        db.close()


def _print_header(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


if __name__ == "__main__":
    raise SystemExit(main())
