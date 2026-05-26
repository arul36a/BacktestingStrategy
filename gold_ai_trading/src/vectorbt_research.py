"""
VectorBT-powered research: fast signal matrices, multi-parameter sweeps,
and portfolio KPI extraction for GOLD strategies.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go

try:
    import vectorbt as vbt
except ImportError as exc:  # pragma: no cover - import guard for optional envs
    vbt = None  # type: ignore
    _VBT_IMPORT_ERR = exc
else:
    _VBT_IMPORT_ERR = None

from .evaluation import summarize_backtest_returns
from .utils import resolve_path

_LOGGER = logging.getLogger(__name__)


def _pandas_long_only_returns_from_signals(
    close: pd.Series,
    entries: pd.Series,
    exits: pd.Series,
    *,
    fee_per_side: float,
    slip_per_side: float,
) -> pd.Series:
    """
    Long-only approximation for research sweeps (no VectorBT dependency).

    PnL for bar ``t`` uses the position carried **into** that bar (`held_before`), then signals
    at the bar close flip state for subsequent bars — similar to reacting at the close before
    the next bar opens (close-only approximation).

    Turnover (−1,0,+1) pays ``fee_per_side + slip_per_side`` per unit change — rough but stable
    vs sweeps across thousands of combos. Confirm edge behavior on Backtrader.
    """
    r = close.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)
    idx = close.index

    ev = entries.reindex(idx).fillna(False).values.astype(np.bool_)
    xv = exits.reindex(idx).fillna(False).values.astype(np.bool_)
    rr = r.values.astype(float)

    held = False
    strat = np.zeros(len(rr))
    txn_cost = fee_per_side + slip_per_side

    for t in range(len(rr)):
        held_before = held
        earn = float(held_before) * rr[t]

        new_held = held_before
        if held_before and xv[t]:
            new_held = False
        elif (not held_before) and ev[t]:
            new_held = True

        turnover = abs(int(new_held) - int(held_before))
        strat[t] = earn - float(turnover) * txn_cost
        held = new_held

    return pd.Series(strat, index=idx, name="return")


class VectorBTGoldResearch:
    """Encapsulates data matrix → signals → portfolios for parameter studies."""

    def __init__(self, config: dict) -> None:
        self.cfg = config
        self.vb_cfg = dict(config.get("vectorbt") or {})
        self._use_vectorbt = vbt is not None
        if not self._use_vectorbt:
            _LOGGER.warning(
                "vectorbt is not installed (common on Python 3.14 while Numba wheels catch up). "
                "Using built-in Pandas research backtester for sweeps. "
                "For full vectorbt, use Python 3.11–3.13 and `pip install vectorbt`.",
            )

    def ema_crossover_signals(self, close: pd.Series, fast: int = 21, slow: int = 55) -> tuple[pd.Series, pd.Series]:
        fast_ma = close.ewm(span=fast, adjust=False).mean()
        slow_ma = close.ewm(span=slow, adjust=False).mean()
        entries = (fast_ma > slow_ma) & (fast_ma.shift(1) <= slow_ma.shift(1))
        exits = (fast_ma < slow_ma) & (fast_ma.shift(1) >= slow_ma.shift(1))
        return entries.astype(bool), exits.astype(bool)

    def rsi_filter_signals(self, rsi: pd.Series, buy_level: float, sell_level: float) -> tuple[pd.Series, pd.Series]:
        entries = (rsi < buy_level) & (rsi.shift(1) >= buy_level)
        exits = (rsi > sell_level) & (rsi.shift(1) <= sell_level)
        return entries.astype(bool), exits.astype(bool)

    def run_portfolio_from_signals(
        self,
        price: pd.Series,
        *,
        entries: pd.Series,
        exits: pd.Series,
        freq: Optional[str] = None,
        init_cash: float = 100_000,
    ) -> "vbt.Portfolio":
        if not self._use_vectorbt or vbt is None:
            raise RuntimeError(
                "VectorBT portfolios require `pip install vectorbt` "
                "(Python 3.11–3.13 recommended). Sweep mode still works via Pandas fallback."
            )
        fees = float(self.vb_cfg.get("fees", 0.00035))
        slippage = float(self.vb_cfg.get("slippage_pct", 0.0001))
        freq = freq or str(self.vb_cfg.get("freq", "1h"))
        pf = vbt.Portfolio.from_signals(
            close=price,
            entries=entries,
            exits=exits,
            freq=freq,
            init_cash=init_cash,
            fees=fees,
            slippage=slippage,
            direction="longonly",
        )
        return pf

    def param_sweep_ema_fast_slow(
        self,
        close: pd.Series,
        fast_grid: Iterable[int],
        slow_grid: Iterable[int],
    ) -> pd.DataFrame:
        """Grid over (fast,slow). Returns dataframe of KPI rows."""
        rows: list[dict[str, Any]] = []
        periods = float(self.vb_cfg.get("periods_per_year", 252 * 6))
        fees = float(self.vb_cfg.get("fees", 0.00035))
        slip = float(self.vb_cfg.get("slippage_pct", 0.0001))

        for f in fast_grid:
            for s in slow_grid:
                if f >= s:
                    continue
                ent, ex = self.ema_crossover_signals(close, fast=f, slow=s)
                try:
                    if self._use_vectorbt and vbt is not None:
                        pf = self.run_portfolio_from_signals(close, entries=ent, exits=ex)
                        rets = pd.Series(pf.returns()).fillna(0)
                        summ = summarize_backtest_returns(rets, periods_per_year=periods)
                        rows.append(
                            {
                                "fast": f,
                                "slow": s,
                                "total_return": float(pf.total_return()),
                                "sharpe_ratio": pf.sharpe_ratio(),
                                "max_drawdown": pf.max_drawdown(),
                                "cagr_estimate": summ.cagr,
                                "win_rate_approx": summ.win_rate if summ.win_rate is not None else np.nan,
                                "engine": "vectorbt",
                            }
                        )
                    else:
                        rets = _pandas_long_only_returns_from_signals(
                            close,
                            ent,
                            ex,
                            fee_per_side=fees,
                            slip_per_side=slip,
                        ).fillna(0)
                        summ = summarize_backtest_returns(rets, periods_per_year=periods)
                        equity = (1 + rets).cumprod()
                        rows.append(
                            {
                                "fast": f,
                                "slow": s,
                                "total_return": float(equity.iloc[-1] - 1.0),
                                "sharpe_ratio": summ.sharpe,
                                "max_drawdown": float(summ.max_drawdown),
                                "cagr_estimate": summ.cagr,
                                "win_rate_approx": summ.win_rate if summ.win_rate is not None else np.nan,
                                "engine": "pandas",
                            }
                        )
                except Exception as exc:  # noqa: PERF203 — research loop diagnostics
                    _LOGGER.warning("param (%s,%s) failed: %s", f, s, exc)
        return pd.DataFrame(rows)

    def heatmap_figure(self, sweep_df: pd.DataFrame, value_col: str = "sharpe_ratio") -> go.Figure:
        """Pivot (fast × slow) and render Plotly heatmap."""
        piv = sweep_df.pivot(index="slow", columns="fast", values=value_col)
        fig = go.Figure(
            data=go.Heatmap(z=piv.values, x=list(piv.columns), y=list(piv.index), colorscale="Viridis"),
        )
        fig.update_layout(title=f"Optimization heatmap: {value_col}", xaxis_title="fast", yaxis_title="slow")
        return fig


def dataframe_from_portfolio(pf: Any) -> tuple[pd.Series, pd.DataFrame]:
    """Equity curve and per-bar returns for dashboards."""
    if vbt is None or pf is None:
        raise RuntimeError("VectorBT Portfolio required.")
    rets = pd.Series(pf.returns()).fillna(0)
    eq = (1 + rets).cumprod()
    return eq, pd.DataFrame({"return": rets})


def save_research_snapshot(out_dir: str, sweep_df: pd.DataFrame, fig: Optional[go.Figure] = None) -> Path:
    root = resolve_path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    path_csv = root / "vectorbt_param_sweep.csv"
    sweep_df.to_csv(path_csv, index=False)
    if fig is not None:
        fig.write_html(str(root / "vectorbt_param_heatmap.html"))
    return root
