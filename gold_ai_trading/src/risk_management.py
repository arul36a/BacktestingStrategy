"""
Risk helpers: fractional Kelly-style sizing placeholders, stop levels from ATR,
and portfolio-level constraint checks wired into Backtrader strategies.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass
class RiskConfig:
    max_position_pct_nav: float
    atr_stop_multiplier: float
    take_profit_rr: float
    use_trailing: bool
    trailing_pct: float
    max_daily_loss_pct: float


def risk_cfg_from_yaml(cfg: Mapping[str, Any]) -> RiskConfig:
    r = cfg.get("risk") or cfg
    return RiskConfig(
        max_position_pct_nav=float(r.get("max_position_pct_nav", 0.25)),
        atr_stop_multiplier=float(r.get("atr_stop_multiplier", 2.0)),
        take_profit_rr=float(r.get("take_profit_rr", 1.5)),
        use_trailing=bool(r.get("use_trailing", True)),
        trailing_pct=float(r.get("trailing_pct", 0.015)),
        max_daily_loss_pct=float(r.get("max_daily_loss_pct", 0.02)),
    )


def atr_stop_price(direction: float, entry: float, atr: float, mult: float) -> float:
    """direction: +1 long, -1 short."""
    if direction > 0:
        return float(entry - mult * atr)
    return float(entry + mult * atr)


def atr_take_profit(direction: float, entry: float, atr: float, mult_sl: float, rr: float) -> float:
    risk_per_unit = mult_sl * atr
    if direction > 0:
        return float(entry + rr * risk_per_unit)
    return float(entry - rr * risk_per_unit)


def fractional_position_units(
    nav: float,
    price: float,
    risk_budget_nav_pct: float,
    stop_distance_price: float,
    contract_multiplier: float = 1.0,
) -> int:
    """
    Position size assuming linear PnL: risk dollars = units * stop_distance * multiplier.

    Returns integer units (floored). For micro lots / forex, reinterpret `contract_multiplier`.
    """
    risk_dollars = max(nav * risk_budget_nav_pct, 0.0)
    per_unit_loss = abs(stop_distance_price * contract_multiplier)
    if per_unit_loss <= 0:
        return 0
    raw = risk_dollars / per_unit_loss
    return max(int(raw), 0)
