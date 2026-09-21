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
    comparison_table,
    monthly_activity_summary,
    run_monte_carlo,
    split_by_period,
    standard_baselines,
    trades_per_month,
)
from bot.backtesting.engine import BacktestResult
from bot.config import ConfigError, LoadedConfig, load_config
from bot.data import (
    BinanceRestProvider,
    CsvProvider,
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
    parser.add_argument(
        "mode",
        choices=("backtest", "paper", "live", "download"),
        help="modo de ejecucion; 'download' solo descarga y cachea velas",
    )
    parser.add_argument("-c", "--config", default=None, help="ruta del settings.yaml")
    parser.add_argument("--symbols", default=None, help="lista separada por comas que sustituye a data.symbols")
    parser.add_argument("--timeframe", default=None, help="temporalidad de las velas (p. ej. 15m)")
    parser.add_argument("--balance", type=float, default=None, help="capital inicial para backtest y paper")
    parser.add_argument("--start", default=None, help="inicio del backtest (YYYY-MM-DD)")
    parser.add_argument("--end", default=None, help="fin del backtest (YYYY-MM-DD)")
    parser.add_argument("--offline", action="store_true", help="usa velas sinteticas en vez de la red")
    parser.add_argument("--csv-dir", default=None, help="lee las velas de CSV en ese directorio en vez de la red")
    parser.add_argument(
        "--kill-switch",
        choices=("resume_next_day", "halt", "both"),
        default=None,
        help="politica del freno en backtest: reanudar al dia siguiente, detenerse, o comparar ambas",
    )
    parser.add_argument(
        "--holdout-months",
        type=int,
        default=None,
        help="meses finales reservados sin evaluar (0 desactiva el holdout)",
    )
    parser.add_argument("--no-cache-refresh", action="store_true", help="no descarga: usa solo la cache local")
    parser.add_argument("--iterations", type=int, default=None, help="numero maximo de ciclos en paper/live")
    parser.add_argument("--walk-forward", action="store_true", help="fuerza el analisis walk-forward")
    parser.add_argument("--monte-carlo", action="store_true", help="fuerza la simulacion de Monte Carlo")
    parser.add_argument("--db", default=None, help="ruta de la base de datos SQLite")
    return parser


def overrides_from_args(args: argparse.Namespace) -> dict[str, Any]:
    """Traduce los argumentos de linea de comandos a un parche de configuracion."""
    # 'download' no es un modo de operativa: no toca execution.mode.
    patch: dict[str, Any] = {}
    if args.mode in ("backtest", "paper", "live"):
        patch["execution"] = {"mode": args.mode}
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
    if args.kill_switch:
        backtesting["kill_switch_policy"] = args.kill_switch
    if args.holdout_months is not None:
        backtesting["holdout_months"] = args.holdout_months
    if backtesting:
        patch["backtesting"] = backtesting

    if args.db:
        patch["logging"] = {"db_path": args.db}
    return patch


def build_provider(
    loaded: LoadedConfig, *, offline: bool, mode: str, csv_dir: str | None = None
) -> MarketDataProvider:
    """Elige la fuente de velas: CSV, sinteticas o el REST publico de Binance."""
    if csv_dir:
        return CsvProvider(csv_dir)
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
            bars = max(1000, min(needed, 60_000))
    return SyntheticProvider(bars=bars, seed=7, anchor=anchor)


# ----------------------------------------------------------------- modos

def _holdout_split(loaded: LoadedConfig, frames) -> tuple[pd.Timestamp | None, pd.Timestamp | None, pd.Timestamp | None]:
    """Devuelve (inicio, fin evaluable, fin total) aplicando el holdout.

    El holdout son los ultimos ``holdout_months`` de historico. No se evaluan:
    existen para que quede un tramo de datos que ninguna decision de ajuste ha
    visto todavia, y que solo tiene valor mientras siga sin mirarse.
    """
    config = loaded.config.backtesting
    start = pd.Timestamp(config.start, tz="UTC") if config.start else None
    last = max(frame["open_time"].iloc[-1] for frame in frames.values() if not frame.empty)
    total_end = pd.Timestamp(config.end, tz="UTC") if config.end else last
    total_end = min(total_end, last)

    if config.holdout_months <= 0:
        return start, total_end, total_end
    return start, total_end - pd.DateOffset(months=config.holdout_months), total_end


def _run_scenarios(
    loaded: LoadedConfig,
    frames,
    *,
    start,
    end,
    db: Database,
    run_id: int,
) -> dict[str, "BacktestResult"]:
    """Ejecuta la politica de kill switch pedida: una de las dos, o ambas."""
    policy = loaded.config.backtesting.kill_switch_policy
    wanted = ("resume_next_day", "halt") if policy == "both" else (policy,)

    results: dict[str, BacktestResult] = {}
    for index, mode in enumerate(wanted):
        # Solo el primer escenario persiste: dos corridas sobre el mismo
        # periodo duplicarian cada trade en la base de datos.
        engine = BacktestEngine(
            loaded,
            db=db if index == 0 else None,
            run_id=run_id if index == 0 else None,
            on_kill_switch=mode,
        )
        results[mode] = engine.run(
            frames, initial_balance=loaded.config.backtesting.initial_balance, start=start, end=end
        )
    return results


def run_backtest(loaded: LoadedConfig, feed: DataFeed, db: Database, run_id: int, args: argparse.Namespace) -> int:
    config = loaded.config.backtesting
    raw_start = pd.Timestamp(config.start, tz="UTC") if config.start else None
    raw_end = pd.Timestamp(config.end, tz="UTC") if config.end else None

    frames = feed.bulk_history(start=raw_start, end=raw_end, refresh=not args.no_cache_refresh)
    start, eval_end, total_end = _holdout_split(loaded, frames)

    _print_header("DATOS")
    for symbol, frame in sorted(frames.items()):
        print(f"  {symbol:10} {len(frame):>7} velas   {frame['open_time'].iloc[0].date()} -> {frame['open_time'].iloc[-1].date()}")
    if config.holdout_months > 0:
        print(
            f"\n  Evaluado: hasta {eval_end.date()}"
            f"\n  HOLDOUT reservado ({config.holdout_months} meses): {eval_end.date()} -> {total_end.date()} — sin evaluar"
        )

    scenarios = _run_scenarios(loaded, frames, start=start, end=eval_end, db=db, run_id=run_id)
    principal = scenarios.get("resume_next_day") or next(iter(scenarios.values()))

    # ------------------------------------------------ escenarios del freno
    _print_header("KILL SWITCH: LOS DOS ESCENARIOS")
    etiquetas = {
        "resume_next_day": "Reanudando al dia siguiente",
        "halt": "Estricto (para hasta revision manual)",
    }
    for mode, result in scenarios.items():
        print(f"\n  {etiquetas.get(mode, mode)}")
        print(f"    {result.metrics.summary()}")
        print(f"    activaciones del freno: {len(result.kill_switch_events)}")
        if result.halted_reason:
            print(f"    DETENIDO en {result.end}: {result.halted_reason}")
            print(f"    periodo cubierto antes de parar: {result.metrics.days:.0f} de {(eval_end - (start or eval_end)).days} dias")

    # ------------------------------------------------------- comparativa
    balance = config.initial_balance
    baselines = standard_baselines(
        {s: _clip_frame(f, start, eval_end) for s, f in frames.items()},
        initial_balance=balance,
        fee_bps=loaded.config.execution.fee_bps,
        slippage_bps=loaded.config.execution.slippage_bps,
        ema_fast=loaded.config.indicators.ema_fast,
        ema_slow=loaded.config.indicators.ema_slow,
    )

    entries = {"BOT (estrategia)": (principal.metrics, principal.metrics.trades)}
    for mode, result in scenarios.items():
        if mode != "resume_next_day":
            entries[f"BOT ({etiquetas[mode]})"] = (result.metrics, result.metrics.trades)
    for baseline in baselines:
        entries[baseline.name] = (baseline.metrics, baseline.trade_count)

    _print_header("COMPARATIVA CONTRA REFERENCIAS")
    print(f"comisiones {loaded.config.execution.fee_bps} bps por lado | deslizamiento {loaded.config.execution.slippage_bps} bps | capital inicial {balance:,.0f}")
    print()
    print(comparison_table(entries).to_string(index=False))

    # -------------------------------------------------- actividad mensual
    _print_header("OPERACIONES POR MES")
    monthly = trades_per_month(principal.trades)
    resumen = monthly_activity_summary(principal.trades)
    print(
        f"{resumen['meses']} meses cubiertos | media {resumen['media']} ops/mes | "
        f"mediana {resumen['mediana']} | maximo {resumen['maximo']} | sin operar {resumen['meses_sin_operar']}"
    )
    if not monthly.empty:
        print()
        print(monthly.to_string(index=False))

    # ------------------------------------------------------- por semestre
    _print_header("RESULTADOS POR SEMESTRE")
    semesters = split_by_period(principal.equity_curve, principal.trades, freq="semester")
    print(semesters.to_string(index=False) if not semesters.empty else "(periodo demasiado corto)")

    # -------------------------------------------------------- rechazos
    counts = principal.rejection_counts()
    if not counts.empty:
        _print_header("POR QUE NO SE OPERO (top 10)")
        print(counts.head(10).to_string())

    # ------------------------------------------------------ Monte Carlo
    if config.monte_carlo.enabled and principal.trades:
        simulation = run_monte_carlo(
            principal.trades, initial_balance=balance,
            iterations=config.monte_carlo.iterations, seed=config.monte_carlo.seed,
        )
        _print_header("MONTE CARLO")
        print(simulation.summary())

    # ------------------------------------------------------ walk-forward
    if config.walk_forward.enabled:
        _print_header("WALK-FORWARD")
        report = WalkForwardAnalysis(
            loaded,
            # Grid pequeno a proposito: cada combinacion es un backtest
            # completo por ventana, y cuanto mas grande es el grid mas facil
            # resulta encontrar parametros que solo funcionan en el pasado.
            grid={
                "scoring.min_score": [55.0, 65.0],
                "risk.stop_atr_multiple": [1.5, 2.2],
            },
        ).run({s: _clip_frame(f, start, eval_end) for s, f in frames.items()})
        print(report.summary())
        frame = report.to_frame()
        if not frame.empty:
            print()
            print(frame.to_string(index=False))
            print(
                "\n  Eficiencia = retorno fuera de muestra / retorno en entrenamiento."
                "\n  Por debajo de ~0.5 los parametros se estan ajustando al pasado."
            )

    db.set_state("last_backtest", {"config_hash": principal.config_hash, **principal.metrics.as_dict()})
    print(f"\nhash de configuracion: {principal.config_hash}")
    return EXIT_OK


def _clip_frame(frame: pd.DataFrame, start, end) -> pd.DataFrame:
    out = frame
    if start is not None:
        out = out[out["open_time"] >= start]
    if end is not None:
        out = out[out["open_time"] <= end]
    return out.reset_index(drop=True)


def run_download(loaded: LoadedConfig, feed: DataFeed, db: Database, run_id: int, args: argparse.Namespace) -> int:
    """Descarga el historico completo y lo deja cacheado en parquet.

    Separar la descarga del backtest permite bajar los datos una vez (que es
    lo lento y lo que depende de la red) y despues iterar sobre la estrategia
    tantas veces como haga falta sin volver a tocar el exchange.
    """
    config = loaded.config.backtesting
    start = pd.Timestamp(config.start, tz="UTC") if config.start else None
    end = pd.Timestamp(config.end, tz="UTC") if config.end else None

    _print_header("DESCARGA DE VELAS")
    print(f"timeframe {loaded.config.data.timeframe} | destino {loaded.config.data.cache_dir}")
    if start:
        print(f"desde {start.date()} hasta {end.date() if end else 'hoy'}\n")

    fallos: list[str] = []
    for symbol in loaded.config.data.symbols:
        try:
            frame = feed.history(symbol, start=start, end=end, refresh=True)
        except Exception as exc:  # noqa: BLE001 - se informa por simbolo y se sigue
            fallos.append(f"{symbol}: {exc}")
            print(f"  {symbol:12} FALLO: {str(exc)[:110]}")
            continue
        stats = feed.stats[symbol]
        cobertura = f"{stats.first.date()} -> {stats.last.date()}" if stats.first is not None else "sin datos"
        print(f"  {symbol:12} {len(frame):>8} velas   {cobertura}   (nuevas: {stats.downloaded})")

    if fallos:
        print(f"\n{len(fallos)} simbolo(s) sin datos utilizables:")
        for fallo in fallos:
            print(f"  - {fallo}")
        return EXIT_RUNTIME
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

    provider = build_provider(loaded, offline=args.offline, mode=args.mode, csv_dir=args.csv_dir)
    feed = DataFeed(provider, loaded.config)
    handlers = {
        "backtest": run_backtest,
        "paper": run_paper,
        "live": run_live,
        "download": run_download,
    }

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
