"""
Feature engineering pipeline: returns, volatility, momentum, lags,
and optional multi-timeframe alignment (no lookahead; higher TF shifted).
"""

from __future__ import annotations

import logging
from typing import Any, Iterable, Mapping, Optional

import numpy as np
import pandas as pd

from .indicators import add_all_indicators

_LOGGER = logging.getLogger(__name__)


def pct_returns(series: pd.Series, horizon: int) -> pd.Series:
    return series.pct_change(periods=horizon)


def rolling_volatility(series: pd.Series, window: int, annual_factor: Optional[float] = None) -> pd.Series:
    r = series.pct_change()
    vol = r.rolling(window).std()
    if annual_factor is None:
        return vol
    return vol * (annual_factor ** 0.5)


def add_return_features(close: pd.Series, horizons: Iterable[int]) -> pd.DataFrame:
    parts = []
    for h in horizons:
        parts.append(pct_returns(close, h).rename(f"ret_{h}"))
    return pd.concat(parts, axis=1)


def add_momentum(close: pd.Series, windows: Iterable[int]) -> pd.DataFrame:
    parts = []
    for w in windows:
        mom = close / close.shift(w) - 1.0
        parts.append(mom.rename(f"mom_{w}"))
    return pd.concat(parts, axis=1)


def add_lagged_features(features: pd.DataFrame, lag_values: Iterable[int]) -> pd.DataFrame:
    out = features.copy()
    for col in features.columns:
        for lag in lag_values:
            out[f"{col}_lag_{lag}"] = features[col].shift(lag)
    return out


def inject_higher_timeframe_signals(
    base_df: pd.DataFrame,
    higher_dfs: Mapping[str, pd.DataFrame],
    prefix: str = "htf_",
    shift_forward: bool = True,
) -> pd.DataFrame:
    """
    For each timeframe string key in `higher_dfs`, reindex closes to base index using
    `ffill()` (no lookahead beyond bar close alignment). Optionally shift(+1 bar) after
    ffill so only **past** closed HTF values affect each base bar.
    """
    out = base_df.copy()
    for tf, hdf in higher_dfs.items():
        if "close" not in hdf.columns:
            raise ValueError(f"higher_dfs['{tf}'] must contain 'close'")
        aligned = hdf["close"].reindex(out.index, method="ffill")
        if shift_forward:
            aligned = aligned.shift(1)
        out[f"{prefix}{tf}_close"] = aligned
        ret = aligned.pct_change()
        out[f"{prefix}{tf}_ret"] = ret
        out[f"{prefix}{tf}_trend_up"] = (aligned > aligned.rolling(20).mean()).astype(float)
    return out


class FeaturePipeline:
    """End-to-end: indicators + engineered features."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        self.cfg = dict(config)

    def transform(self, ohlcv: pd.DataFrame, higher_timeframes: Optional[dict[str, pd.DataFrame]] = None) -> pd.DataFrame:
        df = add_all_indicators(ohlcv, self.cfg)
        feats = pd.DataFrame(index=df.index)

        close = df["close"]
        fg = self.cfg.get("feature_engineering", {})

        ret_h = fg.get("return_horizons") or [1, 5, 10]
        feats = feats.join(add_return_features(close, ret_h))

        vol_w = int(fg.get("volatility_window", 20))
        # Approximate hourly annualization (~ 252 * 24 for regular sessions; adjust for venue)
        annual = 252 * 24
        feats[f"vol_{vol_w}"] = rolling_volatility(close, vol_w, annual_factor=float(annual))
        vol_raw = close.pct_change().rolling(vol_w).std()
        q1 = vol_raw.quantile(0.33)
        q2 = vol_raw.quantile(0.66)
        if len(vol_raw.dropna()):
            feats["vol_regime"] = vol_raw.apply(
                lambda x: np.nan if pd.isna(x) else (0 if x <= q1 else (1 if x <= q2 else 2)),
            )
        else:
            feats["vol_regime"] = np.nan

        mom_w = fg.get("momentum_windows") or [5, 10, 20]
        feats = feats.join(add_momentum(close, mom_w))

        indicator_cols = [c for c in df.columns if c not in {"open", "high", "low", "close", "volume"}]
        ind_block = df[indicator_cols]
        feats = feats.join(ind_block)

        lag_vals = fg.get("lag_features") or [1, 2]
        feats = add_lagged_features(feats, lag_vals)

        if higher_timeframes:
            feats = inject_higher_timeframe_signals(feats, higher_timeframes, prefix="htf_")

        return feats.dropna()


def classification_target_from_returns(forward_returns: pd.Series, threshold: float = 0.0) -> pd.Series:
    return (forward_returns > threshold).astype(int)


def forward_return(close: pd.Series, horizon: int) -> pd.Series:
    """Return from t to t+h (fraction)."""
    return close.shift(-horizon) / close - 1.0


def triple_barrier_labels(
    close: pd.Series,
    *,
    horizon: int,
    atr: Optional[pd.Series] = None,
    tp_atr_mult: float = 1.5,
    sl_atr_mult: float = 1.0,
    min_abs_ret: float = 0.0,
) -> pd.Series:
    """
    Simplified triple-barrier labels.

    Uses future close path over `horizon` bars and ATR-scaled dynamic barriers:
      +1 if take-profit barrier is reached first,
       0 if stop-loss barrier is reached first,
      fallback: final horizon return > min_abs_ret.
    """
    idx = close.index
    c = close.astype(float).values
    if atr is None:
        atr_arr = np.full_like(c, np.nan, dtype=float)
    else:
        atr_arr = atr.reindex(idx).astype(float).values

    labels = np.full(len(c), np.nan, dtype=float)
    for i in range(0, len(c) - horizon):
        entry = c[i]
        if not np.isfinite(entry) or entry == 0:
            continue
        atr_i = atr_arr[i]
        dyn = (abs(atr_i) / entry) if np.isfinite(atr_i) and atr_i > 0 else 0.0
        up = tp_atr_mult * dyn
        dn = sl_atr_mult * dyn

        future = c[i + 1 : i + horizon + 1]
        path_ret = future / entry - 1.0

        up_hits = np.where(path_ret >= up)[0] if up > 0 else np.array([], dtype=int)
        dn_hits = np.where(path_ret <= -dn)[0] if dn > 0 else np.array([], dtype=int)

        first_up = int(up_hits[0]) if len(up_hits) else 10**9
        first_dn = int(dn_hits[0]) if len(dn_hits) else 10**9

        if first_up < first_dn:
            labels[i] = 1.0
        elif first_dn < first_up:
            labels[i] = 0.0
        else:
            labels[i] = 1.0 if path_ret[-1] > min_abs_ret else 0.0

    return pd.Series(labels, index=idx, name="tb_label")


def make_supervised_frames(
    features: pd.DataFrame,
    close: pd.Series,
    *,
    horizon: int,
    mode: str = "classification",
    classification_threshold: float = 0.0,
    label_mode: str = "forward_return",
    atr_series: Optional[pd.Series] = None,
    tb_tp_atr_mult: float = 1.5,
    tb_sl_atr_mult: float = 1.0,
    tb_min_abs_ret: float = 0.0,
    drop_na: bool = True,
) -> tuple[pd.DataFrame, pd.Series]:
    fr = forward_return(close, horizon)
    aligned = features.join(fr.rename("fwd_ret"))
    if mode == "classification" and label_mode == "triple_barrier":
        tb = triple_barrier_labels(
            close,
            horizon=horizon,
            atr=atr_series,
            tp_atr_mult=tb_tp_atr_mult,
            sl_atr_mult=tb_sl_atr_mult,
            min_abs_ret=tb_min_abs_ret,
        )
        aligned = aligned.join(tb)
    if drop_na:
        aligned = aligned.dropna(how="any")
    X = aligned.drop(columns=["fwd_ret"])
    if mode == "classification":
        if label_mode == "triple_barrier" and "tb_label" in aligned.columns:
            y = aligned["tb_label"].astype(float)
            X = X.drop(columns=["tb_label"], errors="ignore")
        else:
            y = classification_target_from_returns(aligned["fwd_ret"], classification_threshold).astype(float)
    elif mode == "regression":
        y = aligned["fwd_ret"].astype(float)
    else:
        raise ValueError("mode must be 'classification' or 'regression'")
    return X.astype(float), y.astype(float)
