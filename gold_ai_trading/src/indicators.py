"""
Technical indicators computed with Pandas/Numpy — no TA-Lib / pandas-ta required,
keeping Python 3.11–3.14 installs lightweight and CI-friendly.

Add pandas-ta or TA-Lib later if you prefer vendor-tuned primitives.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

import numpy as np
import pandas as pd


def rsi(close: pd.Series, length: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(span=length, adjust=False).mean()
    avg_loss = loss.ewm(span=length, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    return out.rename("rsi")


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal_len: int = 9) -> pd.DataFrame:
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    line_macd = ema_fast - ema_slow
    signal = line_macd.ewm(span=signal_len, adjust=False).mean()
    hist = line_macd - signal
    return pd.DataFrame(
        {"macd_line": line_macd, "macd_signal": signal, "macd_hist": hist},
        index=close.index,
    )


def ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False).mean()


def sma(series: pd.Series, length: int) -> pd.Series:
    return series.rolling(length).mean()


def atr(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(span=length, adjust=False).mean().rename("atr")


def bollinger(close: pd.Series, length: int = 20, std: float = 2.0) -> pd.DataFrame:
    mid = close.rolling(length).mean()
    dev = close.rolling(length).std(ddof=0)
    upper = mid + std * dev
    lower = mid - std * dev
    return pd.DataFrame({"bb_low": lower, "bb_mid": mid, "bb_high": upper}, index=close.index)


def supertrend(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    length: int = 10,
    multiplier: float = 3.0,
) -> pd.DataFrame:
    """
    TradingView-style Supertrend (multiplier × ATR) with deterministic bar-by-bar state.
    `supertrend_direction`: +1 long leg, −1 short leg.
    """

    atr_line = atr(high, low, close, length).rename("atr_st")
    hl2 = (high + low) / 2.0
    upper = hl2 + multiplier * atr_line
    lower = hl2 - multiplier * atr_line

    fu = upper.copy().values.astype(float)
    fl = lower.copy().values.astype(float)
    u = upper.values.astype(float)
    l_ = lower.values.astype(float)
    cl = close.values.astype(float)

    n = len(close)
    for i in range(1, n):
        fu[i] = u[i] if cl[i - 1] > fu[i - 1] else min(u[i], fu[i - 1])
        fl[i] = l_[i] if cl[i - 1] < fl[i - 1] else max(l_[i], fl[i - 1])

    direction = np.zeros(n, dtype=np.int8)
    st = np.zeros(n, dtype=float)
    trend_up = False

    for i in range(n):
        up = fu[i]
        lo = fl[i]
        px = cl[i]

        if i == 0:
            trend_up = px >= lo
        else:
            if trend_up:
                trend_up = not (px < lo)
            else:
                trend_up = px > up

        direction[i] = 1 if trend_up else -1
        st[i] = lo if trend_up else up

    return pd.DataFrame(
        {"supertrend_line": pd.Series(st, index=close.index), "supertrend_direction": pd.Series(direction, index=close.index)},
    )


def vwap_daily_anchor(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
) -> pd.Series:
    """Session VWAP resetting each UTC calendar day (anchor configurable later)."""

    typical = (high + low + close) / 3.0
    v = volume.fillna(0).astype(float)
    g = pd.Grouper(freq="D")
    pv = (typical * v).groupby(g).cumsum()
    denom = v.groupby(g).cumsum().replace(0, np.nan)
    return (pv / denom).rename("vwap")


def add_all_indicators(
    df: pd.DataFrame,
    cfg: Optional[Mapping[str, Any]] = None,
) -> pd.DataFrame:
    cfg = dict(cfg or {})
    ind_cfg = cfg.get("indicators") if "indicators" in cfg else cfg

    rsi_len = int(ind_cfg.get("rsi_length", 14))
    m_fast = int(ind_cfg.get("macd_fast", 12))
    m_slow = int(ind_cfg.get("macd_slow", 26))
    m_sig = int(ind_cfg.get("macd_signal", 9))
    ema_fast = int(ind_cfg.get("ema_fast", 21))
    ema_slow = int(ind_cfg.get("ema_slow", 55))
    bb_len = int(ind_cfg.get("bb_length", 20))
    bb_std = float(ind_cfg.get("bb_std", 2))
    atr_len = int(ind_cfg.get("atr_length", 14))
    st_len = int(ind_cfg.get("supertrend_length", 10))
    st_mult = float(ind_cfg.get("supertrend_multiplier", 3.0))

    h = df["high"].astype(float)
    l = df["low"].astype(float)
    c = df["close"].astype(float)
    v = df["volume"].astype(float).clip(lower=0)

    out = df.copy()

    out["rsi"] = rsi(c, rsi_len)

    macd_df = macd(c, fast=m_fast, slow=m_slow, signal_len=m_sig)
    out = out.join(macd_df)

    out["ema_fast"] = ema(c, ema_fast)
    out["ema_slow"] = ema(c, ema_slow)
    out["sma_20"] = sma(c, 20)

    out["atr"] = atr(h, l, c, atr_len)

    bb = bollinger(c, bb_len, bb_std)
    out["bb_low"] = bb["bb_low"]
    out["bb_mid"] = bb["bb_mid"]
    out["bb_high"] = bb["bb_high"]

    out["vwap"] = vwap_daily_anchor(h, l, c, v)

    st = supertrend(h, l, c, length=st_len, multiplier=st_mult)
    out["supertrend"] = st["supertrend_line"]
    out["supertrend_direction"] = st["supertrend_direction"]

    return out
