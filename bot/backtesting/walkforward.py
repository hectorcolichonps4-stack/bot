"""Analisis walk-forward.

Optimizar y evaluar sobre el mismo tramo de historico solo mide la capacidad
de ajustarse al pasado. Aqui cada ventana elige parametros con datos de
entrenamiento y los juzga con datos que no ha visto; lo que cuenta es la
concatenacion de esos tramos fuera de muestra.
"""

from __future__ import annotations

import itertools
import json
import logging
from dataclasses import dataclass, field
from typing import Any, List, Mapping, Sequence

import pandas as pd

from bot.backtesting.engine import BacktestEngine, BacktestResult
from bot.backtesting.metrics import Metrics, compute_metrics
from bot.config.loader import LoadedConfig
from bot.config.schema import Config
from bot.strategy.engine import StrategyEngine

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Window:
    """Un tramo de entrenamiento seguido de su tramo de prueba."""

    index: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


@dataclass
class FoldResult:
    """Resultado de una ventana: parametros elegidos y su comportamiento real."""

    window: Window
    best_params: Mapping[str, Any]
    train_metrics: Metrics
    test_metrics: Metrics
    test_result: BacktestResult


@dataclass
class WalkForwardReport:
    """Conjunto de ventanas mas las metricas agregadas fuera de muestra."""

    folds: List[FoldResult] = field(default_factory=list)
    combined_equity: pd.Series = field(default_factory=lambda: pd.Series(dtype="float64"))
    combined_metrics: Metrics = field(default_factory=Metrics)

    @property
    def efficiency(self) -> float:
        """Rendimiento fuera de muestra dividido entre el de entrenamiento.

        Por debajo de ~0.5 la estrategia esta sobreajustada: rinde en el tramo
        que uso para elegir parametros y se desinfla fuera de el.
        """
        train = sum(f.train_metrics.total_return_pct for f in self.folds)
        test = sum(f.test_metrics.total_return_pct for f in self.folds)
        return test / train if train != 0 else 0.0

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "ventana": f.window.index,
                    "train_desde": f.window.train_start.date(),
                    "train_hasta": f.window.train_end.date(),
                    "test_desde": f.window.test_start.date(),
                    "test_hasta": f.window.test_end.date(),
                    "parametros": ", ".join(f"{k}={v}" for k, v in f.best_params.items()) or "por defecto",
                    "retorno_train_%": round(f.train_metrics.total_return_pct, 2),
                    "retorno_test_%": round(f.test_metrics.total_return_pct, 2),
                    "trades_test": f.test_metrics.trades,
                    "maxDD_test_%": round(f.test_metrics.max_drawdown_pct, 2),
                }
                for f in self.folds
            ]
        )

    def summary(self) -> str:
        if not self.folds:
            return "walk-forward sin ventanas evaluables"
        return (
            f"{len(self.folds)} ventanas | fuera de muestra: {self.combined_metrics.summary()} | "
            f"eficiencia {self.efficiency:.2f}"
        )


def build_windows(
    start: pd.Timestamp,
    end: pd.Timestamp,
    *,
    train_days: int,
    test_days: int,
    step_days: int,
) -> List[Window]:
    """Genera ventanas deslizantes que cubran el periodo disponible."""
    windows: List[Window] = []
    train_start = pd.Timestamp(start)
    index = 0
    while True:
        train_end = train_start + pd.Timedelta(days=train_days)
        test_end = train_end + pd.Timedelta(days=test_days)
        if test_end > end:
            break
        windows.append(Window(index, train_start, train_end, train_end, test_end))
        train_start = train_start + pd.Timedelta(days=step_days)
        index += 1
    return windows


def expand_grid(grid: Mapping[str, Sequence[Any]]) -> List[dict[str, Any]]:
    """Producto cartesiano de un grid de parametros con notacion por puntos."""
    if not grid:
        return [{}]
    keys = list(grid)
    return [dict(zip(keys, values)) for values in itertools.product(*(grid[k] for k in keys))]


def apply_params(loaded: LoadedConfig, params: Mapping[str, Any]) -> LoadedConfig:
    """Reconstruye la configuracion con los parametros indicados.

    Se vuelve a pasar por el validador para que cada combinacion tenga su
    propio hash: dos ventanas con parametros distintos no se mezclan nunca en
    los registros.
    """
    from bot.config.loader import compute_config_hash

    payload = loaded.config.model_dump(mode="json")
    for dotted, value in params.items():
        node = payload
        *path, leaf = dotted.split(".")
        for key in path:
            node = node[key]
        node[leaf] = value

    config = Config.model_validate(payload)
    return LoadedConfig(
        config=config,
        config_hash=compute_config_hash(config),
        source_path=loaded.source_path,
        raw=payload,
    )


class WalkForwardAnalysis:
    """Ejecuta el ciclo entrenar/probar sobre ventanas deslizantes."""

    def __init__(
        self,
        loaded: LoadedConfig,
        *,
        grid: Mapping[str, Sequence[Any]] | None = None,
        objective: str = "expectancy_r",
        min_train_trades: int = 5,
    ) -> None:
        self.loaded = loaded
        self.grid = dict(grid or {})
        self.objective = objective
        self.min_train_trades = min_train_trades
        self._prepared: dict[str, dict[str, pd.DataFrame]] = {}

    def _prepare(self, loaded: LoadedConfig, frames: Mapping[str, pd.DataFrame]) -> Mapping[str, pd.DataFrame]:
        """Indicadores cacheados por configuracion de indicadores.

        Un grid que solo toca umbrales o riesgo no cambia los indicadores, asi
        que recalcularlos en cada ventana y combinacion seria el grueso del
        tiempo de un walk-forward.
        """
        key = json.dumps(loaded.config.indicators.model_dump(mode="json"), sort_keys=True)
        if key not in self._prepared:
            engine = StrategyEngine(loaded)
            self._prepared[key] = {
                symbol: engine.prepare(frame) for symbol, frame in frames.items() if not frame.empty
            }
        return self._prepared[key]

    def run(self, frames: Mapping[str, pd.DataFrame]) -> WalkForwardReport:
        config = self.loaded.config.backtesting.walk_forward
        span_start = min(frame["open_time"].iloc[0] for frame in frames.values() if not frame.empty)
        span_end = max(frame["open_time"].iloc[-1] for frame in frames.values() if not frame.empty)

        windows = build_windows(
            span_start, span_end,
            train_days=config.train_days,
            test_days=config.test_days,
            step_days=config.step_days,
        )
        if not windows:
            log.warning(
                "el historico (%s a %s) no cubre ni una ventana de %d+%d dias",
                span_start.date(), span_end.date(), config.train_days, config.test_days,
            )
            return WalkForwardReport()

        balance = self.loaded.config.backtesting.initial_balance
        folds: List[FoldResult] = []
        equity_segments: List[pd.Series] = []
        running_balance = balance

        for window in windows:
            best_params, train_metrics = self._optimize(frames, window, balance)
            candidate = apply_params(self.loaded, best_params) if best_params else self.loaded

            test = BacktestEngine(candidate).run(
                frames,
                initial_balance=running_balance,
                start=window.test_start,
                end=window.test_end,
                prepared=self._prepare(candidate, frames),
            )
            folds.append(FoldResult(window, best_params, train_metrics, test.metrics, test))

            if not test.equity_curve.empty:
                equity_segments.append(test.equity_curve)
                running_balance = float(test.equity_curve.iloc[-1])

        combined = pd.concat(equity_segments).sort_index() if equity_segments else pd.Series(dtype="float64")
        combined = combined[~combined.index.duplicated(keep="last")]
        all_trades = [t for fold in folds for t in fold.test_result.trades]

        return WalkForwardReport(
            folds=folds,
            combined_equity=combined,
            combined_metrics=compute_metrics(combined, all_trades),
        )

    def _optimize(
        self,
        frames: Mapping[str, pd.DataFrame],
        window: Window,
        balance: float,
    ) -> tuple[dict[str, Any], Metrics]:
        """Elige la combinacion con mejor objetivo en el tramo de entrenamiento."""
        best_params: dict[str, Any] = {}
        best_metrics = Metrics()
        best_score = float("-inf")

        for params in expand_grid(self.grid):
            candidate = apply_params(self.loaded, params) if params else self.loaded
            try:
                result = BacktestEngine(candidate).run(
                    frames,
                    initial_balance=balance,
                    start=window.train_start,
                    end=window.train_end,
                    prepared=self._prepare(candidate, frames),
                )
            except ValueError:
                continue

            # Una combinacion con dos operaciones afortunadas no es evidencia.
            if result.metrics.trades < self.min_train_trades:
                continue

            score = float(getattr(result.metrics, self.objective, 0.0))
            if score > best_score:
                best_score, best_params, best_metrics = score, dict(params), result.metrics

        return best_params, best_metrics
