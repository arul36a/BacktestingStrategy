"""
Backtrader execution layer — brokerage simulation, analyzers, and ML-aware strategy wiring.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Mapping, Optional

import backtrader as bt
import numpy as np
import pandas as pd

from .risk_management import RiskConfig, atr_stop_price, atr_take_profit, fractional_position_units, risk_cfg_from_yaml
from .utils import resolve_path

_LOGGER = logging.getLogger(__name__)


class GoldPandasData(bt.feeds.PandasData):  # type: ignore[misc]
    """Pandas OHLCV feed with optional `ema_fast`, `ema_slow`, `atr`, `ml_prob` columns."""

    lines = ("ema_fast", "ema_slow", "atr", "ml_prob")
    params = (
        ("datetime", None),
        ("open", "open"),
        ("high", "high"),
        ("low", "low"),
        ("close", "close"),
        ("volume", "volume"),
        ("openinterest", None),
        ("ema_fast", "ema_fast"),
        ("ema_slow", "ema_slow"),
        ("atr", "atr"),
        ("ml_prob", "ml_prob"),
    )


class GoldMlStrategy(bt.Strategy):  # type: ignore[misc]
    """
    Long-only template: ML probability + EMA trend filter, ATR stop/TP, optional trailing.

    Extend or subclass for shorting, pyramiding, or session filters.
    """

    params = dict(
        prob_long=0.52,
        confidence_edge=0.02,
        warmup=50,
        risk=None,
        contract_multiplier=1.0,
    )

    def __init__(self) -> None:
        super().__init__()
        self.order: Optional[bt.Order] = None
        self.bar_count = 0
        self.entry_price: Optional[float] = None
        self.peak_price: Optional[float] = None
        risk_arg = self.params.risk
        if isinstance(risk_arg, RiskConfig):
            self.risk = risk_arg
        elif isinstance(risk_arg, dict):
            self.risk = risk_cfg_from_yaml({"risk": risk_arg})
        else:
            self.risk = risk_cfg_from_yaml({"risk": {}})

    def notify_order(self, order: bt.Order) -> None:
        if order.status in (order.Completed, order.Canceled, order.Margin, order.Rejected):
            self.order = None

    def log(self, txt: str) -> None:
        _LOGGER.debug("BT[%s] %s", len(self), txt)

    def next(self) -> None:
        self.bar_count += 1
        if self.bar_count <= int(self.params.warmup) or self.order is not None:
            return

        try:
            prob = float(self.data.ml_prob[0])
            if np.isnan(prob):
                prob = 0.48
        except Exception:
            prob = 0.48

        try:
            ema_fast = float(self.data.ema_fast[0])
            ema_slow = float(self.data.ema_slow[0])
        except Exception:
            ema_fast = ema_slow = float(self.data.close[0])

        try:
            atr_now = float(self.data.atr[0])
            if np.isnan(atr_now):
                atr_now = float(abs(self.data.close[0] - self.data.open[0]) + 1e-9)
        except Exception:
            atr_now = float(abs(self.data.close[0] - self.data.open[0]) + 1e-9)

        trend_ok = ema_fast > ema_slow
        edge = float(self.params.confidence_edge)
        conf_ok = abs(prob - 0.5) >= edge
        long_fire = trend_ok and conf_ok and prob >= float(self.params.prob_long)

        px = float(self.data.close[0])

        # Exit management
        if self.position.size > 0 and self.entry_price is not None and atr_now > 0:
            stop_px = atr_stop_price(1.0, float(self.entry_price), float(atr_now), self.risk.atr_stop_multiplier)
            tp_px = atr_take_profit(
                1.0,
                float(self.entry_price),
                float(atr_now),
                self.risk.atr_stop_multiplier,
                self.risk.take_profit_rr,
            )
            self.peak_price = px if self.peak_price is None else max(self.peak_price, px)
            trailing_stop = self.peak_price * (1 - self.risk.trailing_pct)

            exit_now = px <= stop_px or px >= tp_px
            if self.risk.use_trailing:
                exit_now = exit_now or (px <= trailing_stop and px < self.peak_price * 0.998)

            if exit_now:
                self.order = self.close()
                return

        # Entries
        if self.position.size == 0 and long_fire:
            size = self._position_size(px)
            if size > 0:
                self.order = self.buy(size=size)
                self.entry_price = px
                self.peak_price = px

        # Reset trackers when flat
        if self.position.size == 0:
            self.entry_price = None
            self.peak_price = None

    def _position_size(self, price: float) -> int:
        nav = float(self.broker.getvalue())
        stop_budget = price * 0.002
        stop_budget = max(stop_budget, 1e-6)
        pct = getattr(self.risk, "max_position_pct_nav", 0.25)
        mult = float(self.params.contract_multiplier)
        return fractional_position_units(nav, price, pct, stop_budget, contract_multiplier=mult)


class GoldBacktraderEngine:
    """Runner: broker config, data feed, analyzers, JSON report."""

    def __init__(self, config: dict) -> None:
        self.cfg = config
        self.bt_cfg = dict(config.get("backtrader") or {})

    def run(
        self,
        ohlcv: pd.DataFrame,
        *,
        strategy_cls: type = GoldMlStrategy,
        strategy_kwargs: Optional[Mapping[str, Any]] = None,
        report_path: Optional[str] = None,
    ) -> dict[str, Any]:
        cerebro = bt.Cerebro()
        cerebro.broker.setcash(float(self.bt_cfg.get("initial_cash", 100_000)))
        cerebro.broker.setcommission(commission=float(self.bt_cfg.get("commission", 0.00035)))
        cerebro.broker.set_slippage_perc(perc=float(self.bt_cfg.get("slippage_perc", 0.001)))

        df = ohlcv.copy()
        for col in ("ema_fast", "ema_slow", "atr", "ml_prob"):
            if col not in df.columns:
                df[col] = np.nan
        df["ml_prob"] = df["ml_prob"].fillna(0.45)

        cerebro.adddata(GoldPandasData(dataname=df))

        risk = risk_cfg_from_yaml(self.cfg)
        strat_kw = {
            "prob_long": 0.52,
            "confidence_edge": 0.02,
            "warmup": 50,
            "risk": risk,
            "contract_multiplier": 1.0,
        }
        if strategy_kwargs:
            strat_kw.update(strategy_kwargs)
        cerebro.addstrategy(strategy_cls, **strat_kw)

        cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe", timeframe=bt.TimeFrame.Days, annualize=True)
        cerebro.addanalyzer(bt.analyzers.DrawDown, _name="dd")
        cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")
        cerebro.addanalyzer(bt.analyzers.Returns, _name="rets")

        _LOGGER.info("Backtrader run — %s bars", len(df))
        results = cerebro.run()
        strat = results[0]

        sharpe_analysis = strat.analyzers.sharpe.get_analysis()
        dd_analysis = strat.analyzers.dd.get_analysis()
        trades_analysis = strat.analyzers.trades.get_analysis()

        report: dict[str, Any] = {
            "final_value": float(cerebro.broker.getvalue()),
            "sharpe_ratio": sharpe_analysis.get("sharperatio"),
            "max_drawdown_pct": dd_analysis.get("max", {}).get("drawdown"),
            "trade_stats": trades_analysis,
        }

        if report_path:
            out = resolve_path(report_path)
            out.parent.mkdir(parents=True, exist_ok=True)
            safe = {k: v for k, v in report.items() if k != "trade_stats"}
            safe["trade_stats"] = str(trades_analysis)
            out.write_text(json.dumps(safe, indent=2, default=str), encoding="utf-8")
            _LOGGER.info("Wrote report → %s", out)

        return report


def attach_ml_probabilities(
    ohlcv_with_features: pd.DataFrame,
    model: Any,
    feature_columns: list[str],
) -> pd.Series:
    """
    Vectorized probability column for classification ensembles; maps index to `ml_prob`.
    """
    block = ohlcv_with_features.reindex(columns=feature_columns).astype(float).fillna(0)
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(block.values)[:, 1]
    else:
        preds = model.predict(block.values)
        probs = preds.astype(float)
    return pd.Series(probs, index=ohlcv_with_features.index, name="ml_prob")
