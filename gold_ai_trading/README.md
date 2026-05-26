# Gold AI Trading System

Production-oriented **research → ML → execution validation** stack for GOLD (XAU/USD proxies, gold futures, ETFs) using **VectorBT**, **Backtrader**, and an **ensemble ML** layer (XGBoost, LightGBM, Random Forest, optional LSTM/TensorFlow).

## Architecture

| Layer              | Responsibility |
|--------------------|----------------|
| `src/data_loader`  | Cached OHLCV via Yahoo Finance; swap adapters for ICE/CMC/MCX/vendor APIs |
| `src/indicators`   | RSI, MACD, EMA/SMA, BB, ATR, VWAP, Supertrend (pandas-ta) |
| `src/feature_engineering` | Returns, vol, momentum, lags; multi-timeframe merge without lookahead shift |
| `src/vectorbt_research`   | Rapid sweeps / heatmaps / KPI extraction |
| `src/ml_models`    | Voting ensemble, randomized RF tuning hooks, walk-forward evaluation |
| `src/backtrader_engine` | Slippage, commission, sizing, analyzer exports |
| `dashboard/app.py` | Streamlit UI over CSV/HTML artifacts |

**Important disclaimer:** Markets are risky. Yahoo data is imperfect; COMEX/GC proxies do not replicate every broker spread. Extend `GoldDataLoader` before deployment.

## Prerequisites

- Python **3.11+** (recommended **3.11–3.13** for `vectorbt`; 3.14 users skip that line in `requirements.txt` until Numba ships wheels)
- macOS/Linux recommended (symlink for `models/latest_ensemble.pkl` falls back to copying the file on Windows if symlinks are blocked)
- Indicators are **pure Pandas** (`src/indicators.py`); you may still add TA-Lib/`pandas-ta` as an optional adapter if you rely on their exact tuning

## Setup

```bash
cd gold_ai_trading
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
# Optional GPU / deep-learning path:
# pip install tensorflow-metal  # Apple Silicon tuning
```

## CLI workflows

Train / research artifacts land in `reports/`, persisted models under `models/`.

```bash
# VectorBT EMA crossover sweep (+ heatmap HTML snapshot)
python main.py research --interval 1h
#
# Without vectorbt installed (e.g. Python 3.14 — Numba lag), research uses a lightweight
# pandas sweep backend; CSV + heatmap formats are unchanged. Install vectorbt under 3.11–3.13 for the full stack.

# Ensemble ML + walk-forward diagnostics + Monte Carlo summaries
python main.py train_ml --interval 1h --no-cache   # omit --no-cache to reuse parquet cache

# Backtrader realism pass using newest saved ensemble probs
python main.py validate_bt --interval 1h --prob-long 0.55
```

Dashboard:

```bash
streamlit run dashboard/app.py
```

## Configuration

- `config/default.yaml` — global knobs (symbols, risk, ML, broker sim)
- `config/strategies/example_ema_ml.yaml` — illustrates strategy YAML contract

Override path with `GOLD_AI_CONFIG=/abs/path.yaml`.

### Data sources

| Asset            | Default ticker (Yahoo) | Notes |
|------------------|-------------------------|-------|
| XAU/USD proxy    | `GC=F`                  | COMEX gold future |
| ETF              | `GLD`                   | US ETF |
| MCX gold         | Placeholder in YAML     | Swap to broker historical API |

## Testing

```bash
pytest -q
```

## Project layout

```
gold_ai_trading/
├── config/
├── dashboard/
├── data/            # parquet cache (gitignored by default)
├── models/          # joblib artifacts
├── notebooks/
├── reports/         # CSV/HTML backtest + research exports
├── src/
├── tests/
├── main.py
└── requirements.txt
```

## Optimization & research tips

- Use **coarser bars** (1h/4h/daily) first; microstructure noise dominates 1m results on retail feeds.
- Always **purge** observations around train/test boundaries when labels use overlapping horizons.
- Pair **VectorBT** speed with **Backtrader** realism before committing capital.
- Track **regime-specific** performance (see `vol_regime_series.csv` after training).

## Roadmap (suggested)

1. Swap Yahoo loader for paid historical + level-2 where available
2. Add IBKR / OANDA execution adapters with idempotent order IDs
3. Live **paper** mode with kill-switch + max daily loss enforcement
4. Replace simple vol regime with HMM / changepoint library (ruptures)
5. GPU batch training for large tick datasets (cuDF / Polars integration)

## License

Use at your own risk — no investment advice.
