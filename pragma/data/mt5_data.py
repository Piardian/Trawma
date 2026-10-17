"""MetaTrader 5 data adapter.

Requires a locally installed and logged-in MT5 terminal and the official
MetaTrader5 Python package. MT5 timestamps are UTC; returned indexes are UTC.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Optional

import pandas as pd


class MT5DataError(RuntimeError):
    """Raised when MT5 cannot provide valid market data."""


_TIMEFRAME_NAMES = {
    "M1": "TIMEFRAME_M1",
    "M5": "TIMEFRAME_M5",
    "M15": "TIMEFRAME_M15",
    "M30": "TIMEFRAME_M30",
    "H1": "TIMEFRAME_H1",
    "M60": "TIMEFRAME_H1",
    "H4": "TIMEFRAME_H4",
    "D1": "TIMEFRAME_D1",
}
_TIMEFRAME_DELTAS = {
    "M1": timedelta(minutes=1), "M5": timedelta(minutes=5),
    "M15": timedelta(minutes=15), "M30": timedelta(minutes=30),
    "H1": timedelta(hours=1), "M60": timedelta(hours=1),
    "H4": timedelta(hours=4), "D1": timedelta(days=1),
}


def filter_closed_bars(bars: pd.DataFrame, timeframe: str,
                       now: datetime | pd.Timestamp | None = None) -> pd.DataFrame:
    """Keep only bars whose full timeframe has elapsed.

    MT5 bar timestamps denote bar opening time. This filter works on a
    DatetimeIndex or a `time` column and accepts an injectable UTC clock for
    deterministic tests. It never mutates the input frame.
    """
    key = timeframe.upper()
    if key not in _TIMEFRAME_DELTAS:
        raise ValueError(f"Unsupported timeframe: {timeframe}")
    if bars.empty:
        return bars.copy()
    if isinstance(bars.index, pd.DatetimeIndex):
        times = pd.DatetimeIndex(bars.index)
    elif "time" in bars.columns:
        times = pd.DatetimeIndex(pd.to_datetime(bars["time"], utc=True))
    else:
        raise ValueError("bars must have a DatetimeIndex or a time column")
    clock = pd.Timestamp(now if now is not None else datetime.now(timezone.utc))
    if clock.tzinfo is None:
        clock = clock.tz_localize("UTC")
    else:
        clock = clock.tz_convert("UTC")
    utc_times = times.tz_localize("UTC") if times.tz is None else times.tz_convert("UTC")
    keep = (utc_times + pd.Timedelta(_TIMEFRAME_DELTAS[key])) <= clock
    return bars.loc[keep].copy()


class MT5DataClient:
    """Small, explicit adapter around MT5 rates and ticks APIs."""

    def __init__(self, mt5_module=None):
        if mt5_module is None:
            try:
                import MetaTrader5 as mt5_module  # type: ignore[no-redef]
            except ImportError as exc:
                raise MT5DataError(
                    "MetaTrader5 package is not installed. Install requirements.txt "
                    "and run this on the computer with the MT5 terminal."
                ) from exc
        self.mt5 = mt5_module
        self._initialized_here = False

    def connect(self) -> None:
        if self.mt5.terminal_info() is not None:
            return
        if not self.mt5.initialize():
            raise MT5DataError(f"MT5 initialize failed: {self.mt5.last_error()}")
        self._initialized_here = True

    def close(self) -> None:
        if self._initialized_here:
            self.mt5.shutdown()
            self._initialized_here = False

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def _timeframe(self, timeframe: str):
        name = _TIMEFRAME_NAMES.get(timeframe.upper())
        if not name or not hasattr(self.mt5, name):
            raise ValueError(f"Unsupported timeframe: {timeframe}")
        return getattr(self.mt5, name)

    @staticmethod
    def _normalize_rates(raw) -> pd.DataFrame:
        if raw is None or len(raw) == 0:
            raise MT5DataError("MT5 returned no rates")
        frame = pd.DataFrame(raw)
        if "time" not in frame:
            raise MT5DataError("MT5 rates response has no time field")
        frame["time"] = pd.to_datetime(frame["time"], unit="s", utc=True)
        frame = frame.set_index("time").sort_index()
        frame = frame[~frame.index.duplicated(keep="last")]
        required = {"open", "high", "low", "close", "tick_volume"}
        missing = required.difference(frame.columns)
        if missing:
            raise MT5DataError(f"Rates response missing columns: {sorted(missing)}")
        for col in ("open", "high", "low", "close", "tick_volume"):
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
        frame = frame.dropna(subset=list(required))
        if (frame["high"] < frame["low"]).any():
            raise MT5DataError("Invalid OHLC data: high is below low")
        return frame

    def symbols(self) -> list[str]:
        """Return symbol names exposed by the connected MT5 terminal."""
        self.connect()
        raw = self.mt5.symbols_get()
        if raw is None:
            raise MT5DataError(f"MT5 symbols_get failed: {self.mt5.last_error()}")
        return sorted({
            str(item.name) for item in raw
            if getattr(item, "name", None)
        }, key=str.upper)

    def rates(self, symbol: str, timeframe: str = "M1", count: int = 1500) -> pd.DataFrame:
        if count <= 0:
            raise ValueError("count must be positive")
        self.connect()
        if not self.mt5.symbol_select(symbol, True):
            raise MT5DataError(f"Cannot select symbol {symbol!r}: {self.mt5.last_error()}")
        raw = self.mt5.copy_rates_from_pos(symbol, self._timeframe(timeframe), 0, count)
        return self._normalize_rates(raw)

    def rates_from(self, symbol: str, timeframe: str, start: datetime,
                   count: int) -> pd.DataFrame:
        """Fetch rates ending at/around a UTC-aware datetime."""
        if start.tzinfo is None:
            raise ValueError("start must be timezone-aware")
        self.connect()
        if not self.mt5.symbol_select(symbol, True):
            raise MT5DataError(f"Cannot select symbol {symbol!r}")
        raw = self.mt5.copy_rates_from(
            symbol, self._timeframe(timeframe), start.astimezone(timezone.utc), count
        )
        return self._normalize_rates(raw)

    def ticks_from(self, symbol: str, start: datetime, count: int = 10000) -> pd.DataFrame:
        """Fetch raw ticks from a UTC-aware start time."""
        if start.tzinfo is None:
            raise ValueError("start must be timezone-aware")
        if count <= 0:
            raise ValueError("count must be positive")
        self.connect()
        if not self.mt5.symbol_select(symbol, True):
            raise MT5DataError(f"Cannot select symbol {symbol!r}")
        raw = self.mt5.copy_ticks_from(
            symbol, start.astimezone(timezone.utc), count, self.mt5.COPY_TICKS_ALL
        )
        if raw is None or len(raw) == 0:
            raise MT5DataError(f"No ticks returned for {symbol}: {self.mt5.last_error()}")
        frame = pd.DataFrame(raw)
        time_col = "time_msc" if "time_msc" in frame.columns else "time"
        unit = "ms" if time_col == "time_msc" else "s"
        frame["time"] = pd.to_datetime(frame[time_col], unit=unit, utc=True)
        return frame.set_index("time").sort_index()

    @staticmethod
    def tick_price_histogram(
        ticks: pd.DataFrame,
        bins: int = 100,
        price_field: str | None = None,
        volume_field: str | None = None,
    ) -> pd.DataFrame:
        """Build a tick-price histogram; tick counts are activity, not trade volume."""
        import numpy as np

        if bins < 2:
            raise ValueError("bins must be >= 2")
        if ticks.empty:
            raise ValueError("ticks cannot be empty")
        candidates = [price_field] if price_field else ["last", "bid", "ask"]
        selected_price = next((name for name in candidates
                               if name and name in ticks.columns
                               and (pd.to_numeric(ticks[name], errors="coerce") > 0).any()), None)
        if selected_price is None:
            raise MT5DataError("Tick frame has no usable last/bid/ask price column")
        prices = pd.to_numeric(ticks[selected_price], errors="coerce")
        valid = prices.notna() & np.isfinite(prices) & (prices > 0)
        selected_volume = volume_field
        if selected_volume is None:
            selected_volume = next((name for name in ("volume_real", "volume")
                                    if name in ticks.columns
                                    and (pd.to_numeric(ticks[name], errors="coerce") > 0).any()), None)
        if selected_volume is not None:
            if selected_volume not in ticks.columns:
                raise ValueError(f"volume_field {selected_volume!r} not found")
            weights = pd.to_numeric(ticks[selected_volume], errors="coerce")
            valid &= weights.notna() & np.isfinite(weights) & (weights >= 0)
            weights = weights.fillna(0).to_numpy(dtype=float)
        else:
            weights = np.ones(len(ticks), dtype=float)
        prices_array = prices.to_numpy(dtype=float)
        prices_array = prices_array[valid.to_numpy()]
        weights = weights[valid.to_numpy()]
        if len(prices_array) == 0:
            raise MT5DataError("No valid ticks remain after price/volume validation")
        if float(weights.sum()) <= 0:
            raise MT5DataError("Tick weights sum to zero")
        low, high = float(prices_array.min()), float(prices_array.max())
        if np.isclose(low, high):
            pad = max(abs(low) * 1e-9, np.finfo(float).eps * 16)
            low, high = low - pad, high + pad
        hist, edges = np.histogram(prices_array, bins=bins, range=(low, high), weights=weights)
        return pd.DataFrame({
            "price_low": edges[:-1], "price_high": edges[1:],
            "price": (edges[:-1] + edges[1:]) / 2,
            "tick_weight": hist.astype(float),
        })
