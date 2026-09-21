# Bot de trading

Bot de trading algoritmico para spot de Binance, con el mismo codigo de
estrategia corriendo en backtest, paper y live. Cada modo cambia solo la
implementacion de `Broker` que se inyecta; las reglas no se duplican.

> **Aviso.** Operar con dinero real puede hacerte perder todo el capital.
> El modo `live` esta bloqueado salvo confirmacion explicita en el YAML y,
> por defecto, apunta a la testnet. Usa `backtest` y `paper` durante el
> tiempo suficiente antes de plantearte otra cosa.

## Estructura

| Modulo | Responsabilidad |
| --- | --- |
| `bot/config` | `settings.yaml` + validacion con pydantic; hash de la configuracion |
| `bot/data` | Descarga de velas, cache en parquet, filtros del exchange |
| `bot/indicators` | EMA, RSI, MACD, ATR, volumen relativo, S/R. Funciones puras |
| `bot/strategy` | Filtros obligatorios (gates) + puntuacion -> senal candidata |
| `bot/risk` | Tamano de posicion, limites, correlacion, cooldowns, kill switch |
| `bot/execution` | Interfaz `Broker` con `BacktestBroker`, `PaperBroker`, `BinanceBroker` |
| `bot/backtesting` | Motor por eventos, walk-forward, benchmark, metricas, Monte Carlo |
| `bot/logging` | SQLite: trades, senales, rechazos, errores, estado |
| `bot/dashboard` | Streamlit, solo lectura sobre la base de datos |
| `tests` | Pruebas unitarias, sobre todo de `risk` e `indicators` |
| `main.py` | Orquestador: `backtest` \| `paper` \| `live` |

Todo cuelga de un unico paquete `bot/` a proposito: un directorio `logging/`
en la raiz del repositorio ensombreceria al `logging` de la libreria estandar
y romperia cualquier import que lo use, empezando por pandas.

## Instalacion

```bash
pip install -r requirements.txt
cp .env.example .env     # solo si vas a usar paper/live contra el exchange
```

## Uso

```bash
# Backtest con datos sinteticos, sin tocar la red
python main.py backtest --offline --symbols BTCUSDT,ETHUSDT

# Backtest real con walk-forward y Monte Carlo
python main.py backtest --start 2024-01-01 --end 2024-12-31 --walk-forward --monte-carlo

# Paper trading (precios reales, dinero simulado)
python main.py paper

# Live: exige testnet o confirmacion explicita en settings.yaml
python main.py live

# Dashboard
streamlit run bot/dashboard/app.py -- --db var/bot.sqlite

# Tests
python -m pytest
```

Argumentos utiles: `--config`, `--symbols`, `--timeframe`, `--balance`,
`--start`, `--end`, `--offline`, `--no-cache-refresh`, `--iterations`, `--db`.

## Como se toma una decision

```
velas -> indicadores -> gates (obligatorios) -> puntuacion -> riesgo -> orden
             |              |                       |            |
             |              v                       v            v
             |          rechazo               rechazo       rechazo
             |          (motivo)               (score)     (limite/tamano)
             v
         todo se registra en SQLite
```

1. **Gates.** Filtros binarios: tendencia alineada, volumen relativo minimo,
   liquidez, RSI no extremo, ATR dentro de rango, spread aceptable y recorrido
   libre hasta el siguiente nivel. Si uno falla no se puntua nada.
2. **Puntuacion.** Seis componentes (tendencia, momento, RSI, volumen,
   estructura, volatilidad), cada uno en `[0, 1]`, ponderados y normalizados a
   100 puntos. Cambiar los pesos no cambia la escala de `min_score`.
3. **Riesgo.** Kill switch, duplicados, limites diarios y semanales,
   cooldowns, correlacion con lo ya abierto, tamano de posicion y exposicion
   agregada, en ese orden.

**Cada rechazo se guarda con su motivo.** Es la respuesta a la pregunta que
mas veces se hace uno con un bot: "¿por que no esta operando?". El dashboard
tiene una pestana dedicada.

## Trazabilidad por hash de configuracion

`compute_config_hash` calcula un SHA-256 (16 caracteres) sobre la
configuracion ya validada, no sobre el texto del YAML: comentarios, orden de
claves y valores por defecto escritos explicitamente no alteran el hash, y dos
configuraciones equivalentes comparten huella.

Ese hash viaja con cada senal, cada rechazo y cada trade hasta SQLite, asi que
un resultado historico siempre se puede reatribuir a los parametros exactos
que lo produjeron:

```sql
SELECT config_hash, COUNT(*), SUM(pnl) FROM trades GROUP BY config_hash;
```

## Decisiones de diseno que conviene conocer

- **Sin sesgo de anticipacion.** En el backtest la senal se evalua con la vela
  cerrada y la entrada se ejecuta en la apertura de la siguiente. Los pivotes
  de soporte/resistencia se descartan hasta estar confirmados
  (`sr_pivot_window` velas despues), porque usarlos antes equivaldria a mirar
  al futuro. Hay tests que lo comprueban explicitamente.
- **Si una vela toca stop y objetivo, gana el stop.** Sin datos de tick no se
  puede saber el orden; suponer lo peor evita un backtest mas optimista que la
  realidad.
- **El redondeo de cantidades siempre trunca.** Redondear al alza dejaria la
  posicion por encima del riesgo autorizado.
- **El kill switch no se rearma solo.** En vivo hace falta intervencion
  humana. En backtest, `BacktestEngine(on_kill_switch=...)` permite elegir
  entre `resume_next_day` (por defecto, simula que un operador lo revisa y
  reanuda) y `halt` (corta la corrida para ver donde habria obligado a parar).
- **El walk-forward reporta su eficiencia.** Rendimiento fuera de muestra
  dividido entre el de entrenamiento; por debajo de ~0.5 la estrategia esta
  sobreajustada.
- **`BinanceBroker` deja el stop en el exchange.** Coloca un OCO al abrir, de
  modo que la proteccion sigue viva aunque el proceso del bot se caiga. Si el
  OCO no se puede colocar, la posicion se cierra de inmediato.
- **Las credenciales solo se leen del entorno**, por el nombre de variable que
  indique `settings.yaml`. Nunca se escriben en el YAML ni se registran.

## Modo live

Requiere las dos cosas:

1. `exchange.testnet: false` en `settings.yaml`, y
2. `execution.live_confirmation: I-UNDERSTAND-THE-RISK`

Sin la segunda, el bot se niega a arrancar contra dinero real. Con
`testnet: true` (el valor por defecto) opera contra la testnet de Binance y no
hace falta confirmacion.

## Limitaciones conocidas

- `BinanceBroker` es solo spot: no admite cortos. Con `gates.allow_short: true`
  hay que usar un broker de futuros, que no esta implementado.
- Los backtests usan velas, no el libro de ordenes: el deslizamiento es un
  parametro (`execution.slippage_bps`), no una simulacion de profundidad.
- `SyntheticProvider` (`--offline`) genera series reproducibles para probar el
  circuito completo; no imita la microestructura de un mercado real y sus
  resultados no dicen nada sobre la estrategia.
