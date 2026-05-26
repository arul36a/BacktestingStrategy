"""
Performance analytics: risk metrics, Monte Carlo shuffle of returns,
and simple volatility regime tagging for filtering / reporting.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional

import numpy as np
import pandas as pd


@dataclass
class PerformanceSummary:
    cagr: float
    sharpe: float
    max_drawdown: float
    win_rate: Optional[float]
    expectancy: Optional[float]
    trades: Optional[int]


def max_drawdown(equity_curve: pd.Series) -> float:
    roll_max = equity_curve.cummax()
    dd = equity_curve / roll_max - 1.0
    return float(dd.min())


def sharpe_ratio(returns: pd.Series, periods_per_year: float = 252 * 6) -> float:
    """Default ~ 6 hourly bars/day for rough FX-metal session annualization."""
    r = returns.replace([np.inf, -np.inf], np.nan).dropna()
    if r.std() == 0 or len(r) < 10:
        return 0.0
    mu = r.mean()
    sig = r.std(ddof=0)
    return float(np.sqrt(periods_per_year) * mu / sig)


def sortino_ratio(returns: pd.Series, periods_per_year: float = 252 * 6) -> float:
    r = returns.replace([np.inf, -np.inf], np.nan).dropna()
    downside = r[r < 0]
    ds = downside.std(ddof=0)
    if ds == 0 or len(r) < 10:
        return sharpe_ratio(returns, periods_per_year)
    return float(np.sqrt(periods_per_year) * r.mean() / ds)


def cagr_equity_curve(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    start_val = float(equity.iloc[0])
    end_val = float(equity.iloc[-1])
    if start_val <= 0:
        return 0.0
    years = (equity.index[-1] - equity.index[0]).days / 365.25 if len(equity) > 1 else 1.0
    years = max(years, 1e-6)
    return float((end_val / start_val) ** (1.0 / years) - 1.0)


def trade_stats_from_pnl(series: pd.Series) -> tuple[float, float, int]:
    wins = series[series > 0]
    losses = series[series <= 0]
    n = len(series)
    if n == 0:
        return 0.0, 0.0, 0
    win_rate = len(wins) / n if n else 0.0
    avg_win = wins.mean() if len(wins) else 0.0
    avg_loss = losses.mean() if len(losses) else 0.0
    expectancy = float((win_rate * avg_win + (1 - win_rate) * avg_loss))
    return float(win_rate), expectancy, n


def summarize_backtest_returns(
    returns: pd.Series,
    *,
    trades: Optional[pd.Series] = None,
    periods_per_year: float = 252 * 6,
) -> PerformanceSummary:
    eq = (1 + returns.fillna(0)).cumprod()
    wr, exp, n = (None, None, None)
    if trades is not None:
        wr, exp, n = trade_stats_from_pnl(trades)
    return PerformanceSummary(
        cagr=cagr_equity_curve(eq),
        sharpe=sharpe_ratio(returns, periods_per_year),
        max_drawdown=max_drawdown(eq),
        win_rate=wr,
        expectancy=exp,
        trades=n,
    )


def monte_carlo_returns_bootstrap(
    returns: pd.Series,
    simulations: int = 1000,
    block_size: int = 5,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """
    Circular block bootstrap of returns distribution to visualize path dispersion.
    Not a substitute for transaction-cost aware path simulation.
    """
    rng = np.random.default_rng(seed)
    r = returns.dropna().values
    stats = []
    for _ in range(simulations):
        n = len(r)
        equity = []
        cum = 1.0
        dd_min = 0.0
        peak = 1.0
        while len(equity) < n:
            start = rng.integers(0, max(1, n - block_size))
            chunk = r[start : start + block_size]
            for x in chunk:
                cum *= 1 + x
                peak = max(peak, cum)
                dd_min = min(dd_min, cum / peak - 1)
                equity.append(cum)
        equity = equity[:n]
        stats.append({"final_equity_multiple": cum, "max_drawdown": dd_min})

    return pd.DataFrame(stats)


def regime_from_volatility(
    close_rets: pd.Series,
    window: int,
    *,
    low_q: float = 0.33,
    mid_q: float = 0.66,
) -> pd.Series:
    """
    Labels 0/1/2 = low/med/high volatility regime based on rolling realized vol quantiles.

    Stateless over full sample for research dashboards; production should roll quantiles online.
    """
    vol = close_rets.rolling(window).std()
    qs = vol.quantile([low_q, mid_q])
    def _lbl(x: float) -> float:
        if pd.isna(x):
            return np.nan
        if x <= qs.iloc[0]:
            return 0
        if x <= qs.iloc[1]:
            return 1
        return 2
    return vol.map(_lbl).rename("vol_regime")
