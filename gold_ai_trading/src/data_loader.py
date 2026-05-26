"""
Historical market data ingestion and caching for GOLD instruments.

Default provider: Yahoo Finance via `yfinance` (offline cache to Parquet).
MCX commodities require plugging in your vendor (TrueData / Kite / etc.).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional

import pandas as pd
import yfinance as yf

from .utils import project_root, resolve_path

_LOGGER = logging.getLogger(__name__)

_TIMEFRAME_MAP = {
    "1m": "1m",
    "5m": "5m",
    "15m": "15m",
    "1h": "60m",
    "4h": "240m",  # yfinance aggregates from intraday intervals with limitations
    "1d": "1d",
}

_YAHOO_MAX_LOOKBACK_DAYS = {
    "1m": 7,
    "2m": 60,
    "5m": 60,
    "15m": 60,
    "30m": 60,
    "60m": 730,
    "90m": 60,
    "1h": 730,
}


@dataclass
class InstrumentSpec:
    """Logical instrument description for GOLD research."""

    ticker: str
    name: str
    venue: str  # COMEX | ETF | MCX_PLACEHOLDER


class GoldDataLoader:
    """
    Download OHLCV, cache locally, optionally resample to multiple timeframes.

    Notes:
    - Intraday history length is limited by Yahoo (e.g., 730 days max for minute data).
    - For institutional depth, swap `download()` with ICE/CME/vendor REST/gRPC loaders.
    """

    def __init__(self, config: dict) -> None:
        self.cfg = config
        data_cfg = config.get("data", {})
        paths = config.get("paths", {})
        root = project_root()

        self._data_dir = resolve_path(paths.get("data_dir", "./data"), root)
        self._cache_format = str(data_cfg.get("cache_format", "parquet"))
        self._default_symbol = str(data_cfg.get("default_symbol", "GC=F"))

        self._data_dir.mkdir(parents=True, exist_ok=True)

    def cache_path(self, symbol: str, interval: str) -> str:
        safe = symbol.replace("/", "_").replace("=", "")
        suf = ".parquet" if self._cache_format == "parquet" else ".csv"
        return str(self._data_dir / f"{safe}_{interval}{suf}")

    def download(
        self,
        symbol: Optional[str] = None,
        interval: Optional[str] = None,
        start: Optional[str] = None,
        end: Optional[str] = None,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """
        Fetch OHLCV for `symbol` at Yahoo interval mapped from config timeframe labels.

        `interval`: one of keys in `_TIMEFRAME_MAP`, e.g. '1h', '1d'. For 4h, we download
        1h bars and resample externally via `to_timeframe()` for fidelity.
        """
        data_cfg = self.cfg.get("data", {})
        symbol = symbol or data_cfg.get("default_symbol") or self._default_symbol
        timeframe = interval or data_cfg.get("timeframe", "1h")
        iv = _TIMEFRAME_MAP.get(timeframe, timeframe)
        start = start or data_cfg.get("start_date") or "2018-01-01"
        end = data_cfg.get("end_date") or None
        if end == "null":
            end = None
        start = self._sanitize_yahoo_intraday_start(start=start, interval=iv, end=end)

        cache_file = self.cache_path(symbol, iv)
        if use_cache and pd.io.common.file_exists(cache_file):
            df = pd.read_parquet(cache_file) if cache_file.endswith(".parquet") else pd.read_csv(
                cache_file, index_col=0, parse_dates=True
            )
            _LOGGER.info("Loaded cached data: %s rows from %s", len(df), cache_file)
            return self._normalize_ohlcv(df)

        _LOGGER.info(
            "Downloading %s [%s] from %s end=%s", symbol, iv, start, end or "latest"
        )
        df = self._yahoo_download(symbol=symbol, interval=iv, start=start, end=end)
        df = self._normalize_ohlcv(df)

        if use_cache and len(df):
            self._write_cache(cache_file, df)
        return df

    def _sanitize_yahoo_intraday_start(self, start: str, interval: str, end: Optional[str]) -> str:
        """
        Clamp intraday lookback to Yahoo limits to avoid hard failures like:
        "The requested range must be within the last 730 days."
        """
        max_days = _YAHOO_MAX_LOOKBACK_DAYS.get(interval)
        if max_days is None:
            return start

        end_dt = pd.Timestamp.utcnow() if end is None else pd.Timestamp(end)
        if end_dt.tzinfo is None:
            end_dt = end_dt.tz_localize("UTC")
        else:
            end_dt = end_dt.tz_convert("UTC")

        # Keep a safety margin below the hard provider boundary.
        min_allowed = (end_dt - timedelta(days=max_days - 1)).normalize()
        start_dt = pd.Timestamp(start)
        if start_dt.tzinfo is None:
            start_dt = start_dt.tz_localize("UTC")
        else:
            start_dt = start_dt.tz_convert("UTC")

        if start_dt < min_allowed:
            clamped = min_allowed.strftime("%Y-%m-%d")
            _LOGGER.warning(
                "Yahoo interval %s supports ~%s days; clamping start from %s to %s",
                interval,
                max_days,
                start,
                clamped,
            )
            return clamped
        return start_dt.strftime("%Y-%m-%d")

    def _yahoo_download(self, symbol: str, interval: str, start: str, end: Optional[str]) -> pd.DataFrame:
        try:
            raw = yf.download(
                tickers=symbol,
                interval=interval,
                start=start,
                end=end,
                progress=False,
                auto_adjust=False,
                threads=False,
                group_by="column",
            )
        except Exception as exc:
            _LOGGER.exception("yfinance failed for %s: %s", symbol, exc)
            raise RuntimeError(f"Data download failed for {symbol}") from exc

        if isinstance(raw, pd.Series) or raw.empty:
            return pd.DataFrame()

        df = raw.copy()

        if isinstance(df.columns, pd.MultiIndex):
            try:
                df = df.droplevel(1, axis=1).copy()
            except Exception:
                df.columns = pd.Index(str(t[0]) for t in df.columns.to_list())

        cols = [str(c).lower().replace(" ", "_") for c in df.columns]
        flat = pd.DataFrame(df.values, columns=cols, index=df.index)

        rename = {
            "adj close": "adj_close",
            "adj_close": "adj_close",
            "vol": "volume",
        }
        flat = flat.rename(columns={k: v for k, v in rename.items() if k in flat.columns})
        flat.index = pd.DatetimeIndex(flat.index)
        if flat.index.tz is None:
            flat.index = flat.index.tz_localize("UTC")
        else:
            flat.index = flat.index.tz_convert("UTC")
        return flat

    @staticmethod
    def _normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        out = df.copy()
        cols = {"open": "open", "high": "high", "low": "low", "close": "close", "volume": "volume"}
        for canonical in cols:
            if canonical not in out.columns:
                raise ValueError(f"Missing column '{canonical}' in OHLCV frame.")

        # Force lexicographically sorted index after concat/resample scenarios
        out = out.sort_index()
        for c in ["open", "high", "low", "close"]:
            out[c] = out[c].astype(float)
        if "volume" in out.columns:
            out["volume"] = out["volume"].fillna(0).astype(float)
        else:
            out["volume"] = 1.0
        return out

    def _write_cache(self, path: str, df: pd.DataFrame) -> None:
        if path.endswith(".parquet"):
            df.to_parquet(path)
        else:
            df.to_csv(path)
        _LOGGER.info("Wrote cache: %s (%s bars)", path, len(df))

    def to_timeframe(self, df: pd.DataFrame, rule: str) -> pd.DataFrame:
        """Resample bars to pandas offset alias: e.g. '4h', '1D'."""
        o = df["open"].resample(rule).first()
        h = df["high"].resample(rule).max()
        l = df["low"].resample(rule).min()
        cl = df["close"].resample(rule).last()
        v = df["volume"].resample(rule).sum()
        out = pd.concat([o, h, l, cl, v], axis=1)
        out.columns = ["open", "high", "low", "close", "volume"]
        return out.dropna(how="any")
