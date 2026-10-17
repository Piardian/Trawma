"""Binance USD-M Futures OHLCV adapter with true taker-flow delta and CSV cache.

Public endpoints only; no API key or third-party exchange package is required.
The adapter returns the same UTC DatetimeIndex/OHLC/tick_volume shape as MT5DataClient.
"""
from __future__ import annotations

from datetime import datetime, timezone
import csv
import io
import json
import sys
import time
import zipfile
from pathlib import Path
from typing import Callable
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd

FUTURES_URL = "https://fapi.binance.com/fapi/v1/klines"
SPOT_URL = "https://api.binance.com/api/v3/klines"
_INTERVALS = {"M1": "1m", "M5": "5m", "M15": "15m", "M30": "30m",
              "H1": "1h", "M60": "1h", "H4": "4h", "D1": "1d"}
_INTERVAL_MS = {"1m": 60_000, "5m": 300_000, "15m": 900_000,
                "30m": 1_800_000, "1h": 3_600_000, "4h": 14_400_000,
                "1d": 86_400_000}
_COLUMNS = ["open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_asset_volume", "number_of_trades",
            "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"]


class BinanceDataError(RuntimeError):
    """Raised when Binance returns invalid or insufficient market data."""


def btc_leader_allows_direction(btc_regime: str | None, symbol: str, direction: str) -> bool:
    """Return False for counter-BTC altcoin directions, requiring strict IMBALANCE regimes for altcoin trades."""
    pair = normalize_binance_symbol(symbol)
    side = str(direction or "").upper()
    if pair == "BTCUSDT":
        return True
    if not btc_regime:
        return False
    regime = str(btc_regime).upper()
    if side == "LONG":
        return regime == "IMBALANCE_BULL"
    if side == "SHORT":
        return regime == "IMBALANCE_BEAR"
    return False


def normalize_binance_symbol(symbol: str) -> str:
    value = str(symbol or "").strip().upper().replace("/", "").replace("_", "").replace("-", "")
    aliases = {"BTC": "BTCUSDT", "BTCUSD": "BTCUSDT", "XBTUSD": "BTCUSDT",
               "ETH": "ETHUSDT", "ETHUSD": "ETHUSDT", "SOL": "SOLUSDT",
               "SOLUSD": "SOLUSDT", "LINK": "LINKUSDT", "LINKUSD": "LINKUSDT"}
    return aliases.get(value, value)


def _parse_rows(rows: list) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows, columns=_COLUMNS)
    for col in ("open", "high", "low", "close", "volume", "quote_asset_volume",
                "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume"):
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame["time"] = pd.to_datetime(pd.to_numeric(frame["open_time"]), unit="ms", utc=True)
    frame["close_time"] = pd.to_datetime(pd.to_numeric(frame["close_time"]), unit="ms", utc=True)
    frame["buy_volume"] = frame["taker_buy_base_asset_volume"]
    frame["sell_volume"] = frame["volume"] - frame["buy_volume"]
    frame["volume_delta"] = frame["buy_volume"] - frame["sell_volume"]
    # Preserve quote volume separately; tick_volume uses base-asset volume for
    # consistent volume-climax calculations across symbols.
    frame["tick_volume"] = frame["volume"]
    frame = frame.set_index("time").sort_index()
    frame.index.name = "time"
    frame = frame[~frame.index.duplicated(keep="last")]
    return frame


def filter_closed_bars(bars: pd.DataFrame, timeframe: str,
                       now: datetime | pd.Timestamp | None = None) -> pd.DataFrame:
    """Exclude any candle whose exchange close_time has not elapsed."""
    if bars.empty:
        return bars.copy()
    clock = pd.Timestamp(now if now is not None else datetime.now(timezone.utc))
    clock = clock.tz_localize("UTC") if clock.tzinfo is None else clock.tz_convert("UTC")
    if "close_time" in bars.columns:
        close_times = pd.to_datetime(bars["close_time"], utc=True)
        return bars.loc[close_times <= clock].copy()
    key = str(timeframe).upper()
    if key not in _INTERVALS:
        raise ValueError(f"Unsupported timeframe: {timeframe}")
    starts = pd.DatetimeIndex(bars.index)
    starts = starts.tz_localize("UTC") if starts.tz is None else starts.tz_convert("UTC")
    elapsed = pd.to_timedelta(_INTERVAL_MS[_INTERVALS[key]], unit="ms")
    # Binance close_time is inclusive (open + duration - 1ms).
    return bars.loc[starts + elapsed <= clock].copy()


class BinanceDataClient:
    """Paginated, public Binance Futures/Spot klines client with incremental CSV cache."""

    def __init__(self, cache_dir: str | Path = "data_cache", *, no_cache: bool = False,
                 futures_only: bool = False, timeout: float = 15.0, sleep_seconds: float = 0.15,
                 opener: Callable | None = None):
        self.cache_dir = Path(cache_dir)
        self.no_cache = bool(no_cache)
        self.futures_only = bool(futures_only)
        self.timeout = float(timeout)
        self.sleep_seconds = max(0.0, float(sleep_seconds))
        self._opener = opener or urlopen

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    @staticmethod
    def _cache_key(symbol: str, timeframe: str) -> str:
        return f"binance_{normalize_binance_symbol(symbol)}_{str(timeframe).upper()}.csv"

    def _read_cache(self, path: Path) -> pd.DataFrame:
        if self.no_cache or not path.exists():
            return pd.DataFrame()
        try:
            frame = pd.read_csv(path, parse_dates=["time", "close_time"])
            frame["time"] = pd.to_datetime(frame["time"], utc=True)
            frame["close_time"] = pd.to_datetime(frame["close_time"], utc=True)
            frame = frame.set_index("time").sort_index()
            frame.index.name = "time"
            return frame[~frame.index.duplicated(keep="last")]
        except (OSError, ValueError, KeyError, pd.errors.ParserError):
            return pd.DataFrame()

    def _write_cache(self, path: Path, frame: pd.DataFrame) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        saved = frame.reset_index()
        saved.to_csv(tmp, index=False)
        tmp.replace(path)

    def _request_page(self, base_url: str, symbol: str, interval: str,
                      end_time: int, limit: int) -> list:
        params = urlencode({"symbol": symbol, "interval": interval,
                            "endTime": end_time, "limit": limit})
        request = Request(f"{base_url}?{params}", headers={"User-Agent": "Pragma/1.0"})
        max_retries = 3
        for attempt in range(max_retries):
            try:
                with self._opener(request, timeout=self.timeout) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                if isinstance(payload, dict):
                    msg = payload.get("msg", payload)
                    raise BinanceDataError(f"Binance API error: {msg}")
                if not isinstance(payload, list):
                    raise BinanceDataError("Unexpected Binance response format")
                return payload
            except HTTPError as exc:
                if exc.code in (429, 418, 500, 502, 503, 504) and attempt < max_retries - 1:
                    time.sleep(1.0 * (attempt + 1))
                    continue
                raise
            except (URLError, TimeoutError, OSError) as exc:
                if attempt < max_retries - 1:
                    time.sleep(0.5 * (attempt + 1))
                    continue
                raise

    def _fetch_before(self, symbol: str, interval: str, end_time: int,
                      limit: int, *, force_spot: bool = False) -> tuple[pd.DataFrame, str]:
        urls = [SPOT_URL] if force_spot else ([FUTURES_URL] if self.futures_only else [FUTURES_URL, SPOT_URL])
        errors = []
        for url in urls:
            try:
                rows = self._request_page(url, symbol, interval, end_time, min(limit, 1500 if url == FUTURES_URL else 1000))
                return _parse_rows(rows), url
            except (HTTPError, URLError, TimeoutError, OSError, ValueError, BinanceDataError) as exc:
                errors.append(f"{url}: {exc}")
        raise BinanceDataError("All Binance endpoints failed: " + " | ".join(errors))

    def _rates_from_vision_archive(self, symbol: str, interval: str, count: int,
                                   timeframe: str) -> pd.DataFrame:
        """Fallback to Binance's official USD-M Futures public data archive (HTTP 451 regions)."""
        now = pd.Timestamp.now(tz="UTC")
        interval_ms = _INTERVAL_MS[interval]
        start = now - pd.Timedelta(milliseconds=count * interval_ms)
        current_month = now.normalize().replace(day=1)
        frames: list[pd.DataFrame] = []

        def fetch_archive(url: str, max_retries: int = 3) -> pd.DataFrame:
            for attempt in range(max_retries):
                try:
                    request = Request(url, headers={"User-Agent": "Pragma/1.0"})
                    with self._opener(request, timeout=max(self.timeout, 30.0)) as response:
                        payload = response.read()
                    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
                        csv_names = [name for name in archive.namelist() if name.lower().endswith(".csv")]
                        if not csv_names:
                            return pd.DataFrame()
                        with archive.open(csv_names[0]) as stream:
                            rows = list(csv.reader(io.TextIOWrapper(stream, encoding="utf-8")))
                    if rows and rows[0] and rows[0][0].strip().lower() in {"open_time", "open time"}:
                        rows = rows[1:]
                    return _parse_rows(rows)
                except HTTPError as exc:
                    if exc.code == 404:
                        return pd.DataFrame()
                    if attempt < max_retries - 1:
                        time.sleep(1.0 * (attempt + 1))
                        continue
                    raise
                except (URLError, TimeoutError, OSError):
                    if attempt < max_retries - 1:
                        time.sleep(1.0 * (attempt + 1))
                        continue
                    raise
            return pd.DataFrame()

        sys.stderr.write(f"[BINANCE-ARCHIVE] Fetching {symbol} {timeframe} ({count} bars) from data.binance.vision...\n")
        sys.stderr.flush()

        # Binance publishes completed months as monthly archives. The current
        # month is fetched day-by-day because its monthly archive is not final.
        cursor = start.normalize().replace(day=1)
        while cursor < current_month:
            month = cursor.strftime("%Y-%m")
            url = (f"https://data.binance.vision/data/futures/um/monthly/klines/"
                   f"{symbol}/{interval}/{symbol}-{interval}-{month}.zip")
            try:
                frame = fetch_archive(url)
                if not frame.empty:
                    frames.append(frame)
            except HTTPError as exc:
                if exc.code != 404:
                    raise
            cursor = cursor + pd.offsets.MonthBegin(1)

        day = current_month
        while day <= now.normalize():
            date = day.strftime("%Y-%m-%d")
            url = (f"https://data.binance.vision/data/futures/um/daily/klines/"
                   f"{symbol}/{interval}/{symbol}-{interval}-{date}.zip")
            try:
                frame = fetch_archive(url)
                if not frame.empty:
                    frames.append(frame)
            except HTTPError as exc:
                # Today's archive may not be published until the UTC day closes.
                if exc.code != 404:
                    raise
            day += pd.Timedelta(days=1)

        if not frames:
            raise BinanceDataError(f"No Binance USD-M Futures archive data returned for {symbol} {timeframe}")
        result = pd.concat(frames).sort_index()
        result = result[~result.index.duplicated(keep="last")]
        result = filter_closed_bars(result, timeframe)
        if len(result) < count:
            raise BinanceDataError(
                f"Insufficient USD-M Futures archive history for {symbol} {timeframe}: "
                f"{len(result)} closed bars, requested {count}"
            )
        result = result.tail(count).copy()
        result.attrs["source"] = "binance_usdm_futures_archive"
        if not self.no_cache:
            cache_path = self.cache_dir / self._cache_key(symbol, timeframe)
            self._write_cache(cache_path, result)
        sys.stderr.write(f"[BINANCE-ARCHIVE] Completed {symbol} {timeframe}: {len(result)} closed bars.\n")
        sys.stderr.flush()
        return result

    def rates(self, symbol: str, timeframe: str = "M1", count: int = 1500) -> pd.DataFrame:
        if count <= 0:
            raise ValueError("count must be positive")
        tf = str(timeframe).upper()
        if tf not in _INTERVALS:
            raise ValueError(f"Unsupported timeframe: {timeframe}")
        symbol, interval = normalize_binance_symbol(symbol), _INTERVALS[tf]
        if not symbol.endswith("USDT"):
            raise ValueError(f"Binance USD-M Futures symbol must be USDT-quoted: {symbol}")
        cache_path = self.cache_dir / self._cache_key(symbol, tf)
        cached = self._read_cache(cache_path)
        now_ms = int(pd.Timestamp.now(tz="UTC").timestamp() * 1000)
        interval_ms = _INTERVAL_MS[interval]

        frames: list[pd.DataFrame] = []
        try:
            if not cached.empty and not self.no_cache:
                frames.append(cached)
                latest_open_ms = int(cached.index[-1].timestamp() * 1000)
                # 1. Forward sync: fetch new bars up to now
                if latest_open_ms + interval_ms < now_ms:
                    gap_bars = max(2, int((now_ms - latest_open_ms) // interval_ms) + 2)
                    new_page, _ = self._fetch_before(symbol, interval, now_ms, min(1500, gap_bars))
                    if not new_page.empty:
                        frames.append(new_page)
                curr = pd.concat(frames).sort_index()
                curr = curr[~curr.index.duplicated(keep="last")]
                closed_curr = filter_closed_bars(curr, tf)
                # 2. Backward backfill: if requested count > available closed bars
                if len(closed_curr) < count:
                    needed = count - len(closed_curr) + 2
                    oldest_ms = int(curr.index[0].timestamp() * 1000)
                    end_time = oldest_ms - 1
                    while needed > 0:
                        page_limit = min(needed, 1500)
                        page, _ = self._fetch_before(symbol, interval, end_time, page_limit)
                        if page.empty:
                            break
                        frames.append(page)
                        oldest_ms = int(page.index[0].timestamp() * 1000)
                        needed -= len(page)
                        if oldest_ms <= 0 or len(page) < min(page_limit, 1000):
                            break
                        end_time = oldest_ms - 1
                        if self.sleep_seconds and needed > 0:
                            time.sleep(self.sleep_seconds)
            else:
                # Cold fetch: page backward from now until at least count + 2 closed bars collected
                remaining = count + 2
                end_time = now_ms
                while remaining > 0:
                    page_limit = min(remaining, 1500)
                    page, _ = self._fetch_before(symbol, interval, end_time, page_limit)
                    if page.empty:
                        break
                    frames.append(page)
                    oldest_ms = int(page.index[0].timestamp() * 1000)
                    remaining -= len(page)
                    if oldest_ms <= 0 or len(page) < min(page_limit, 1000):
                        break
                    end_time = oldest_ms - 1
                    if self.sleep_seconds and remaining > 0:
                        time.sleep(self.sleep_seconds)


        except BinanceDataError as exc:
            if not cached.empty and len(filter_closed_bars(cached, tf)) >= count:
                result = filter_closed_bars(cached, tf).tail(count).copy()
                result.attrs["source"] = "binance_cache"
                return result
            if self.futures_only and "HTTP Error 451" in str(exc):
                return self._rates_from_vision_archive(symbol, interval, count, tf)
            raise

        if not frames:
            raise BinanceDataError(f"No Binance klines returned for {symbol} {tf}")
        fetched = pd.concat(frames).sort_index()
        fetched = fetched[~fetched.index.duplicated(keep="last")]
        # Closed-bar filter is based on exchange close_time, not local candle assumptions.
        fetched = filter_closed_bars(fetched, tf)
        if not self.no_cache:
            self._write_cache(cache_path, fetched)
        result = fetched.tail(count).copy()
        result.attrs["source"] = "binance"
        return result
