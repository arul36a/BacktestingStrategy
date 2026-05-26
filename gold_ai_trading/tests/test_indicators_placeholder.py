from __future__ import annotations

import pandas as pd


def test_add_all_indicators_shape():
    from src.indicators import add_all_indicators

    idx = pd.date_range("2024-01-01", periods=160, freq="h", tz="UTC")
    sample = pd.DataFrame(
        {
            "open": 2000 + pd.Series(range(len(idx))),
            "high": 2001 + pd.Series(range(len(idx))),
            "low": 1999 + pd.Series(range(len(idx))),
            "close": 2000 + pd.Series(range(len(idx))),
            "volume": 1000 + pd.Series(range(len(idx))),
        },
        index=idx,
    )
    out = add_all_indicators(sample, {})
    assert "rsi" in out.columns
