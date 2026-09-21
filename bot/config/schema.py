"""Esquema de configuracion validado con pydantic.

Todos los modelos son inmutables y rechazan claves desconocidas: un typo en
``settings.yaml`` es un error de arranque, no un parametro silenciosamente
ignorado.
"""

from __future__ import annotations

from datetime import date
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

Timeframe = Literal["1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "12h", "1d"]


class Base(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ExchangeConfig(Base):
    name: Literal["binance"] = "binance"
    testnet: bool = True
    base_url: Optional[str] = None
    api_key_env: str = "BINANCE_API_KEY"
    api_secret_env: str = "BINANCE_API_SECRET"
    recv_window_ms: int = Field(5000, ge=1000, le=60000)
    quote_asset: str = "USDT"


class DataConfig(Base):
    cache_dir: str = "var/cache"
    timeframe: Timeframe = "15m"
    warmup_bars: int = Field(300, ge=50)
    symbols: List[str] = Field(default_factory=lambda: ["BTCUSDT", "ETHUSDT"])
    max_bars_per_request: int = Field(1000, ge=100, le=1500)

    @model_validator(mode="after")
    def _unique_symbols(self) -> "DataConfig":
        if len(set(self.symbols)) != len(self.symbols):
            raise ValueError("data.symbols contiene duplicados")
        if not self.symbols:
            raise ValueError("data.symbols no puede estar vacio")
        return self


class MacdConfig(Base):
    fast: int = Field(12, ge=2)
    slow: int = Field(26, ge=3)
    signal: int = Field(9, ge=2)

    @model_validator(mode="after")
    def _ordered(self) -> "MacdConfig":
        if self.fast >= self.slow:
            raise ValueError("macd.fast debe ser menor que macd.slow")
        return self


class IndicatorsConfig(Base):
    ema_fast: int = Field(20, ge=2)
    ema_slow: int = Field(50, ge=3)
    ema_trend: int = Field(200, ge=10)
    rsi_period: int = Field(14, ge=2)
    macd: MacdConfig = MacdConfig()
    atr_period: int = Field(14, ge=2)
    relative_volume_lookback: int = Field(20, ge=5)
    sr_lookback: int = Field(120, ge=20)
    sr_pivot_window: int = Field(3, ge=1)
    sr_tolerance_pct: float = Field(0.4, gt=0, le=10)

    @model_validator(mode="after")
    def _ordered(self) -> "IndicatorsConfig":
        if not self.ema_fast < self.ema_slow < self.ema_trend:
            raise ValueError("se requiere ema_fast < ema_slow < ema_trend")
        return self


class GatesConfig(Base):
    """Filtros obligatorios. Si uno falla, no hay senal: no se puntua nada."""

    enabled: bool = True
    require_trend_alignment: bool = True
    allow_short: bool = False
    min_relative_volume: float = Field(1.2, gt=0)
    rsi_long_max: float = Field(72.0, gt=0, le=100)
    rsi_short_min: float = Field(28.0, ge=0, lt=100)
    min_atr_pct: float = Field(0.3, ge=0)
    max_atr_pct: float = Field(8.0, gt=0)
    max_spread_pct: float = Field(0.15, gt=0)
    min_quote_volume: float = Field(2_000_000.0, ge=0)
    min_bars_available: int = Field(250, ge=50)
    blacklist: List[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _ranges(self) -> "GatesConfig":
        if self.min_atr_pct >= self.max_atr_pct:
            raise ValueError("gates.min_atr_pct debe ser menor que max_atr_pct")
        return self


class ScoringConfig(Base):
    """Pesos de cada componente. Se normalizan a 100 puntos."""

    min_score: float = Field(60.0, ge=0, le=100)
    weights: Dict[str, float] = Field(
        default_factory=lambda: {
            "trend": 25.0,
            "momentum": 20.0,
            "rsi": 15.0,
            "volume": 15.0,
            "structure": 15.0,
            "volatility": 10.0,
        }
    )

    @model_validator(mode="after")
    def _weights_valid(self) -> "ScoringConfig":
        if not self.weights:
            raise ValueError("scoring.weights no puede estar vacio")
        if any(w < 0 for w in self.weights.values()):
            raise ValueError("scoring.weights no admite pesos negativos")
        if sum(self.weights.values()) <= 0:
            raise ValueError("scoring.weights debe sumar mas de 0")
        return self


class KillSwitchConfig(Base):
    enabled: bool = True
    consecutive_losses: int = Field(4, ge=1)
    max_drawdown_pct: float = Field(15.0, gt=0, le=100)
    max_errors_per_hour: int = Field(10, ge=1)


class RiskConfig(Base):
    spot_only: bool = True
    """Candado de seguridad: solo largos, sin margen y sin apalancamiento.

    Con este valor en ``true`` (el de por defecto) la configuracion rechaza
    ``gates.allow_short`` y los brokers simulados se niegan a abrir cortos, de
    modo que operar en corto no depende de acordarse de dejar una bandera
    apagada: hay que desactivar el candado a proposito.
    """

    risk_per_trade_pct: float = Field(0.75, gt=0, le=10)
    max_position_pct: float = Field(20.0, gt=0, le=100)
    max_concurrent_positions: int = Field(3, ge=1)
    max_exposure_pct: float = Field(60.0, gt=0, le=100)
    max_daily_loss_pct: float = Field(3.0, gt=0, le=100)
    max_weekly_loss_pct: float = Field(7.0, gt=0, le=100)
    max_daily_trades: int = Field(8, ge=1)
    cooldown_minutes_after_loss: int = Field(60, ge=0)
    cooldown_minutes_same_symbol: int = Field(120, ge=0)
    max_correlation: float = Field(0.85, gt=0, le=1)
    correlation_lookback: int = Field(120, ge=20)
    stop_atr_multiple: float = Field(1.8, gt=0)
    take_profit_r_multiple: float = Field(2.2, gt=0)
    kill_switch: KillSwitchConfig = KillSwitchConfig()

    @model_validator(mode="after")
    def _coherent(self) -> "RiskConfig":
        if self.max_daily_loss_pct > self.max_weekly_loss_pct:
            raise ValueError("risk.max_daily_loss_pct no puede superar max_weekly_loss_pct")
        if self.spot_only and self.max_position_pct > 100.0:
            raise ValueError("risk.max_position_pct > 100% implicaria apalancamiento")
        if self.spot_only and self.max_exposure_pct > 100.0:
            raise ValueError("risk.max_exposure_pct > 100% implicaria apalancamiento")
        return self


class ExecutionConfig(Base):
    mode: Literal["backtest", "paper", "live"] = "paper"
    order_type: Literal["market", "limit"] = "market"
    limit_offset_bps: float = Field(5.0, ge=0)
    slippage_bps: float = Field(4.0, ge=0)
    fee_bps: float = Field(7.5, ge=0)
    poll_seconds: int = Field(30, ge=1)
    live_confirmation: str = ""
    """Debe valer exactamente ``I-UNDERSTAND-THE-RISK`` para operar en real."""


class WalkForwardConfig(Base):
    enabled: bool = False
    train_days: int = Field(90, ge=7)
    test_days: int = Field(30, ge=1)
    step_days: int = Field(30, ge=1)


class MonteCarloConfig(Base):
    enabled: bool = False
    iterations: int = Field(1000, ge=10, le=100_000)
    seed: int = 7


KillSwitchPolicy = Literal["resume_next_day", "halt", "both"]


class BacktestConfig(Base):
    start: Optional[date] = None
    end: Optional[date] = None
    initial_balance: float = Field(10_000.0, gt=0)
    benchmark_symbol: str = "BTCUSDT"
    holdout_months: int = Field(0, ge=0, le=60)
    """Meses finales reservados como holdout: no se evaluan hasta pedirlo."""

    kill_switch_policy: KillSwitchPolicy = "both"
    """``halt`` detiene la corrida hasta revision manual, ``resume_next_day``
    reanuda al dia siguiente y ``both`` ejecuta y compara los dos escenarios."""

    walk_forward: WalkForwardConfig = WalkForwardConfig()
    monte_carlo: MonteCarloConfig = MonteCarloConfig()

    @model_validator(mode="after")
    def _range(self) -> "BacktestConfig":
        if self.start and self.end and self.start >= self.end:
            raise ValueError("backtest.start debe ser anterior a backtest.end")
        return self


class LoggingConfig(Base):
    db_path: str = "var/bot.sqlite"
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_rejections: bool = True
    console: bool = True


class Config(Base):
    """Raiz del arbol de configuracion."""

    exchange: ExchangeConfig = ExchangeConfig()
    data: DataConfig = DataConfig()
    indicators: IndicatorsConfig = IndicatorsConfig()
    gates: GatesConfig = GatesConfig()
    scoring: ScoringConfig = ScoringConfig()
    risk: RiskConfig = RiskConfig()
    execution: ExecutionConfig = ExecutionConfig()
    backtesting: BacktestConfig = BacktestConfig()
    logging: LoggingConfig = LoggingConfig()

    @model_validator(mode="after")
    def _spot_only_forbids_shorts(self) -> "Config":
        """El candado de spot no se puede contradecir desde otra seccion."""
        if self.risk.spot_only and self.gates.allow_short:
            raise ValueError(
                "gates.allow_short: true es incompatible con risk.spot_only: true. "
                "El spot no permite vender lo que no se tiene; para operar en corto "
                "harian falta futuros o margen, que este bot no implementa."
            )
        return self

    @model_validator(mode="after")
    def _warmup_covers_indicators(self) -> "Config":
        needed = max(
            self.indicators.ema_trend,
            self.indicators.sr_lookback,
            self.indicators.macd.slow + self.indicators.macd.signal,
            self.gates.min_bars_available,
        )
        if self.data.warmup_bars < needed:
            raise ValueError(
                f"data.warmup_bars={self.data.warmup_bars} es insuficiente; "
                f"los indicadores configurados necesitan al menos {needed} velas"
            )
        return self
