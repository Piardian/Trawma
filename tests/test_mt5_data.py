from datetime import datetime, timezone
from types import SimpleNamespace

import pandas as pd
import pytest

from pragma.data.mt5_data import MT5DataClient, MT5DataError, filter_closed_bars


def test_mt5_closed_bar_filter_uses_utc_bar_open_time():
    bars = pd.DataFrame(
        {"open": [1, 2], "high": [2, 3], "low": [0, 1], "close": [1, 2]},
        index=pd.to_datetime(["2026-01-01T11:59:00Z", "2026-01-01T12:00:00Z"]),
    )
    out = filter_closed_bars(bars, "M1", now=pd.Timestamp("2026-01-01T12:01:00Z"))
    assert len(out) == 2
    out = filter_closed_bars(bars, "M5", now=pd.Timestamp("2026-01-01T12:01:00Z"))
    assert len(out) == 1


def test_mt5_client_rates_and_copy_ticks_from():
    rates = [{
        "time": 1760000000, "open": 1.1, "high": 1.2, "low": 1.0,
        "close": 1.15, "tick_volume": 50, "spread": 2, "real_volume": 0,
    }]
    ticks = [{
        "time": 1760000000, "time_msc": 1760000000123,
        "bid": 1.1, "ask": 1.11, "last": 1.105,
        "volume": 1, "volume_real": 1, "flags": 0,
    }]
    mock = SimpleNamespace(
        terminal_info=lambda: object(), initialize=lambda: True,
        shutdown=lambda: None, last_error=lambda: "mock",
        symbol_select=lambda symbol, selected: True,
        copy_rates_from_pos=lambda *args: rates,
        copy_ticks_from=lambda *args: ticks,
        TIMEFRAME_M1=1, COPY_TICKS_ALL=0,
    )
    with MT5DataClient(mock) as client:
        frame = client.rates("XAUUSD", "M1", count=1)
        tick_frame = client.ticks_from("XAUUSD", datetime(2025, 1, 1, tzinfo=timezone.utc), count=1)
    assert frame.iloc[0]["tick_volume"] == 50
    assert frame.index.tz is not None
    assert tick_frame.iloc[0]["last"] == pytest.approx(1.105)
    assert tick_frame.index.tz is not None


def test_mt5_client_fails_closed_on_missing_rates():
    mock = SimpleNamespace(
        terminal_info=lambda: object(), symbol_select=lambda *args: True,
        copy_rates_from_pos=lambda *args: None, last_error=lambda: "no rates",
        TIMEFRAME_M1=1,
    )
    client = MT5DataClient(mock)
    with pytest.raises(MT5DataError, match="no rates"):
        client.rates("XAUUSD", "M1", count=10)
