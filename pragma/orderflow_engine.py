"""Order-flow features, liquidity pools, and trigger evaluation for Trawma.

Important data note:
- `volume_delta` is treated as exchange-provided aggressor delta only when supplied by
  a source that defines it that way.
- Tick-rule delta is an estimate from tick price changes.
- OHLCV BVC delta is a bar-location proxy, not true bid/ask trade delta.

Input DataFrames should use a DatetimeIndex or a timestamp column named `timestamp`.
All session boundaries are UTC calendar-day boundaries unless a session column is
explicitly supplied.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Literal, Mapping, Sequence

import numpy as np
import pandas as pd


EngineMode = Literal["amt", "orderflow", "hybrid"]


@dataclass(frozen=True)
class OrderFlowConfig:
    session_timezone: str = "UTC"
    session_reset: Literal["daily", "none"] = "daily"
    delta_lookback: int = 20
    atr_period: int = 14
    volume_sma_period: int = 20
    swing_left: int = 5
    swing_right: int = 5
    pool_lookback: int = 100
    volume_spike_multiple: float = 2.2
    narrow_body_atr_multiple: float = 0.25
    poi_atr_multiplier: float = 0.25
    divergence_lookback: int = 15
    sweep_lookback: int = 3
    absorption_delta_multiple: float = 1.5
    flip_delta_multiple: float = 0.8
    # If True, inputs with no usable volume source raise instead of silently
    # synthesizing zero delta.
    require_delta_source: bool = True


@dataclass(frozen=True)
class LiquidityPool:
    kind: str
    side: Literal["LONG", "SHORT", "BOTH"]
    level: float
    lower: float
    upper: float
    timestamp: Any
    source: str
    strength: float = 0.0
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TriggerDecision:
    action: Literal["EXECUTE", "NO_TRADE"]
    side: Literal["LONG", "SHORT", "NONE"] = "NONE"
    reason: str = ""
    metrics: Mapping[str, Any] = field(default_factory=dict)

    @property
    def execute(self) -> bool:
        return self.action == "EXECUTE"


def _normalise_ohlcv(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with normalized lower-case OHLCV names and datetime index."""
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame")
    df = frame.copy()
    rename: dict[Any, str] = {}
    for c in df.columns:
        key = str(c).strip().lower().replace(" ", "_")
        if key in {"open", "high", "low", "close", "volume", "tick_volume",
                   "volume_delta", "timestamp", "time", "session", "bid", "ask",
                   "price", "tick_price", "tick_volume_delta"}:
            rename[c] = key
    df = df.rename(columns=rename)
    if "timestamp" in df.columns:
        idx = pd.to_datetime(df.pop("timestamp"), utc=True, errors="coerce")
        df.index = idx
    elif "time" in df.columns and not isinstance(df.index, pd.DatetimeIndex):
        idx = pd.to_datetime(df.pop("time"), utc=True, errors="coerce")
        df.index = idx
    if not isinstance(df.index, pd.DatetimeIndex):
        # A RangeIndex is allowed for unit tests/features; sessions then use row groups.
        df.index = pd.RangeIndex(len(df), name=df.index.name or "bar")
    elif df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")
    missing = {"open", "high", "low", "close"} - set(df.columns)
    if missing:
        raise ValueError(f"Missing OHLC columns: {sorted(missing)}")
    for c in ("open", "high", "low", "close", "volume", "tick_volume",
              "volume_delta", "bid", "ask", "price", "tick_price"):
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _session_keys(index: pd.Index, df: pd.DataFrame, timezone: str) -> pd.Series:
    if "session" in df.columns:
        return df["session"].astype("string").fillna("UNKNOWN")
    if isinstance(index, pd.DatetimeIndex):
        return pd.Series(index.tz_convert(timezone).date, index=df.index, dtype="object")
    return pd.Series(np.zeros(len(df), dtype=int), index=df.index)


def _true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    return pd.concat(
        [(df["high"] - df["low"]).abs(),
         (df["high"] - prev_close).abs(),
         (df["low"] - prev_close).abs()],
        axis=1,
    ).max(axis=1)


def calculate_atr(frame: pd.DataFrame, period: int = 14) -> pd.Series:
    df = _normalise_ohlcv(frame)
    return _true_range(df).rolling(max(1, int(period)), min_periods=1).mean().rename("atr_14")


def _tick_rule_bar_delta(ticks: pd.DataFrame, bars: pd.DataFrame) -> pd.Series:
    """Aggregate tick-rule signed tick volume into bars using right-closed bar times.

    Ticks must have either DatetimeIndex or a timestamp/time column, plus price (or
    tick_price) and optionally volume. With no tick volume, each tick contributes one.
    """
    t = ticks.copy()
    ren = {}
    for c in t.columns:
        k = str(c).strip().lower().replace(" ", "_")
        if k in {"timestamp", "time", "price", "tick_price", "volume", "tick_volume"}:
            ren[c] = k
    t = t.rename(columns=ren)
    if "timestamp" in t.columns:
        t.index = pd.to_datetime(t.pop("timestamp"), utc=True, errors="coerce")
    elif "time" in t.columns and not isinstance(t.index, pd.DatetimeIndex):
        t.index = pd.to_datetime(t.pop("time"), utc=True, errors="coerce")
    if not isinstance(t.index, pd.DatetimeIndex):
        raise ValueError("ticks requires a DatetimeIndex or timestamp/time column")
    if t.index.tz is None:
        t.index = t.index.tz_localize("UTC")
    else:
        t.index = t.index.tz_convert("UTC")
    price_col = "price" if "price" in t else "tick_price" if "tick_price" in t else None
    if price_col is None:
        raise ValueError("ticks requires price or tick_price")
    t = t.sort_index()
    price = pd.to_numeric(t[price_col], errors="coerce")
    changes = price.diff()
    sign = np.sign(changes)
    # Zero price changes inherit previous non-zero tick direction, defaulting neutral.
    sign = pd.Series(sign, index=t.index).replace(0, np.nan).ffill().fillna(0.0)
    volume_col = "volume" if "volume" in t else "tick_volume" if "tick_volume" in t else None
    vol = pd.to_numeric(t[volume_col], errors="coerce").fillna(1.0) if volume_col else pd.Series(1.0, index=t.index)
    signed = sign * vol
    bar_index = bars.index
    if not isinstance(bar_index, pd.DatetimeIndex):
        raise ValueError("Tick aggregation requires timestamped bars")
    # Map each tick to the most recent bar open at or before the tick time.
    positions = bar_index.get_indexer(t.index, method="pad")
    valid = positions >= 0
    sums = np.zeros(len(bars), dtype=float)
    np.add.at(sums, positions[valid], signed.to_numpy(dtype=float)[valid])
    return pd.Series(sums, index=bars.index, name="bar_delta")


def calculate_orderflow_features(
    frame: pd.DataFrame,
    *,
    ticks: pd.DataFrame | None = None,
    config: OrderFlowConfig | None = None,
) -> pd.DataFrame:
    """Calculate bar delta, session CVD, and delta-scale features.

    Priority: supplied volume_delta -> tick rule from ticks -> BVC from OHLC and volume.
    BVC formula uses tick_volume preferentially, then volume. It is explicitly synthetic.
    """
    cfg = config or OrderFlowConfig()
    df = _normalise_ohlcv(frame)
    if df.empty:
        for name in ("bar_delta", "cvd", "delta_abs_mean_20", "delta_zscore", "atr_14",
                     "vol_sma_20", "delta_source"):
            df[name] = pd.Series(index=df.index, dtype="object" if name == "delta_source" else float)
        return df

    if "volume_delta" in df and df["volume_delta"].notna().any():
        delta = df["volume_delta"].astype(float).fillna(0.0)
        source = "exchange_volume_delta"
    elif ticks is not None:
        delta = _tick_rule_bar_delta(ticks, df)
        source = "tick_rule_estimate"
    else:
        vol_col = "tick_volume" if "tick_volume" in df else "volume" if "volume" in df else None
        if vol_col is None:
            if cfg.require_delta_source:
                raise ValueError("No delta source: need volume_delta, ticks, tick_volume, or volume")
            delta = pd.Series(0.0, index=df.index)
            source = "unavailable_zero"
        else:
            volume = df[vol_col].fillna(0.0).clip(lower=0.0)
            spread = df["high"] - df["low"]
            safe_spread = spread.where(spread != 0, np.nan)
            bvc = volume * (2.0 * (df["close"] - df["low"]) / safe_spread - 1.0)
            previous_close = df["close"].shift(1)
            flat_sign = np.where(df["close"] >= previous_close, 1.0, -1.0)
            flat_sign[0] = 1.0
            delta = bvc.where(spread > 0, volume * flat_sign).replace([np.inf, -np.inf], np.nan).fillna(0.0)
            source = f"bvc_synthetic_{vol_col}"

    df["bar_delta"] = delta.astype(float)
    sessions = _session_keys(df.index, df, cfg.session_timezone)
    if cfg.session_reset == "daily":
        df["cvd"] = df["bar_delta"].groupby(sessions, sort=False).cumsum()
    else:
        df["cvd"] = df["bar_delta"].cumsum()
    df["delta_abs_mean_20"] = df["bar_delta"].abs().rolling(
        max(1, cfg.delta_lookback), min_periods=1
    ).mean()
    mean_delta = df["bar_delta"].rolling(max(1, cfg.delta_lookback), min_periods=1).mean()
    std_delta = df["bar_delta"].rolling(max(1, cfg.delta_lookback), min_periods=2).std(ddof=0)
    df["delta_zscore"] = ((df["bar_delta"] - mean_delta) / std_delta.replace(0.0, np.nan)).fillna(0.0)
    df["atr_14"] = _true_range(df).rolling(max(1, cfg.atr_period), min_periods=1).mean()
    volume_col = "tick_volume" if "tick_volume" in df else "volume" if "volume" in df else None
    if volume_col:
        df["vol_sma_20"] = df[volume_col].rolling(max(1, cfg.volume_sma_period), min_periods=1).mean()
    else:
        df["vol_sma_20"] = np.nan
    df["delta_source"] = source
    return df


def detect_orderflow_pools(
    frame: pd.DataFrame,
    *,
    config: OrderFlowConfig | None = None,
    profile: Any | None = None,
    previous_session_levels: Mapping[str, float] | None = None,
) -> list[LiquidityPool]:
    """Detect confirmed swing liquidity and effort/result absorption candidate pools.

    A pivot is only emitted after its right-hand confirmation bars have closed, avoiding
    use of an unconfirmed future pivot at the current bar.
    """
    cfg = config or OrderFlowConfig()
    df = calculate_orderflow_features(frame, config=cfg)
    if df.empty:
        return []
    atr_mult = cfg.poi_atr_multiplier
    if profile is not None:
        atr_mult = float(getattr(profile, "poi_atr_multiplier", atr_mult))
    pools: list[LiquidityPool] = []
    left, right = max(1, cfg.swing_left), max(1, cfg.swing_right)
    start = max(left, len(df) - cfg.pool_lookback)
    end = len(df) - right
    for i in range(start, max(start, end)):
        lo = df["low"].iloc[i]
        hi = df["high"].iloc[i]
        left_lows = df["low"].iloc[i-left:i]
        right_lows = df["low"].iloc[i+1:i+right+1]
        left_highs = df["high"].iloc[i-left:i]
        right_highs = df["high"].iloc[i+1:i+right+1]
        atr = float(df["atr_14"].iloc[i]) if pd.notna(df["atr_14"].iloc[i]) else 0.0
        half_width = max(0.0, atr_mult * atr)
        if len(left_lows) == left and len(right_lows) == right and lo <= left_lows.min() and lo < right_lows.min():
            pools.append(LiquidityPool("SWING_LOW", "LONG", float(lo), float(lo-half_width),
                                       float(lo+half_width), df.index[i], "confirmed_swing",
                                       metadata={"pivot_index": i}))
        if len(left_highs) == left and len(right_highs) == right and hi >= left_highs.max() and hi > right_highs.max():
            pools.append(LiquidityPool("SWING_HIGH", "SHORT", float(hi), float(hi-half_width),
                                       float(hi+half_width), df.index[i], "confirmed_swing",
                                       metadata={"pivot_index": i}))
    # Previous-session extrema supplied by caller, or derive the last completed session.
    levels = dict(previous_session_levels or {})
    if not levels and len(df) > 1:
        sessions = _session_keys(df.index, df, cfg.session_timezone)
        unique = list(pd.unique(sessions))
        if len(unique) >= 2:
            prev = df.loc[sessions == unique[-2]]
            if not prev.empty:
                levels = {"Session High": float(prev["high"].max()), "Session Low": float(prev["low"].min())}
    for key, val in levels.items():
        if not np.isfinite(float(val)):
            continue
        side: Literal["LONG", "SHORT", "BOTH"] = "LONG" if "low" in key.lower() or "val" in key.lower() else "SHORT" if "high" in key.lower() or "vah" in key.lower() else "BOTH"
        atr = float(df["atr_14"].iloc[-1]) if pd.notna(df["atr_14"].iloc[-1]) else 0.0
        w = max(0.0, atr_mult * atr)
        pools.append(LiquidityPool("SESSION_EXTREME", side, float(val), float(val-w), float(val+w),
                                   df.index[-1], "previous_session", metadata={"level_name": key}))
    # Effort vs result: abnormal volume but small candle body relative to ATR.
    recent_start = max(0, len(df) - min(100, cfg.pool_lookback))
    recent = df.iloc[recent_start:]
    mask = (
        recent["vol_sma_20"].notna()
        & (recent.get("tick_volume", recent.get("volume", pd.Series(np.nan, index=recent.index)))
           >= cfg.volume_spike_multiple * recent["vol_sma_20"])
        & ((recent["close"] - recent["open"]).abs()
           <= cfg.narrow_body_atr_multiple * recent["atr_14"].replace(0.0, np.nan))
    )
    for idx, row in recent.loc[mask].iterrows():
        atr = float(row["atr_14"]) if pd.notna(row["atr_14"]) else 0.0
        w = max(0.0, atr_mult * atr)
        level = float((row["high"] + row["low"]) / 2.0)
        # Candidate is direction-neutral until trigger context classifies it.
        pools.append(LiquidityPool("ABSORPTION_NODE", "BOTH", level, level-w, level+w, idx,
                                   "high_volume_narrow_body",
                                   strength=float(row.get("tick_volume", row.get("volume", 0.0))),
                                   metadata={"bar_delta": float(row["bar_delta"]),
                                             "cvd": float(row["cvd"])}))
    return pools


def _divergence_context(df: pd.DataFrame, side: str, cfg: OrderFlowConfig,
                        poi: LiquidityPool | Mapping[str, Any] | None) -> tuple[str | None, int | None, dict[str, Any]]:
    """Return divergence type, absorption bar position, and diagnostic context."""
    n = len(df)
    if n < 3:
        return None, None, {}
    lookback = min(cfg.divergence_lookback, n - 1)
    prior_end = n - 2  # don't use trigger bar as the absorption extremum
    prior_start = max(0, prior_end - lookback)
    prior = df.iloc[prior_start:prior_end]
    if prior.empty:
        return None, None, {}
    current = df.iloc[-2]  # candidate absorption on prior closed bar
    trigger = df.iloc[-1]  # latest closed bar used for delta flip
    if side == "LONG":
        t1_rel = int(np.argmin(prior["low"].to_numpy()))
        t1_pos = prior_start + t1_rel
        low1, cvd1 = float(df["low"].iloc[t1_pos]), float(df["cvd"].iloc[t1_pos])
        low2, cvd2 = float(current["low"]), float(current["cvd"])
        exhaustion = low2 <= low1 and cvd2 > cvd1
        poi_lower = None
        if isinstance(poi, LiquidityPool):
            poi_lower = poi.lower
        elif isinstance(poi, Mapping):
            poi_lower = poi.get("lower", poi.get("low"))
        closes_above = float(current["close"]) > low1
        if poi_lower is not None:
            closes_above = closes_above or float(current["close"]) > float(poi_lower)
        deltas = df["bar_delta"].iloc[max(0, n - 4):n - 1]
        heavy_sell = bool((deltas < -cfg.absorption_delta_multiple * df["delta_abs_mean_20"].iloc[max(0, n - 4):n - 1]).any())
        absorption = (cvd2 < cvd1 or heavy_sell) and closes_above
    else:
        t1_rel = int(np.argmax(prior["high"].to_numpy()))
        t1_pos = prior_start + t1_rel
        high1, cvd1 = float(df["high"].iloc[t1_pos]), float(df["cvd"].iloc[t1_pos])
        high2, cvd2 = float(current["high"]), float(current["cvd"])
        exhaustion = high2 >= high1 and cvd2 < cvd1
        poi_upper = None
        if isinstance(poi, LiquidityPool):
            poi_upper = poi.upper
        elif isinstance(poi, Mapping):
            poi_upper = poi.get("upper", poi.get("high"))
        closes_below = float(current["close"]) < high1
        if poi_upper is not None:
            closes_below = closes_below or float(current["close"]) < float(poi_upper)
        deltas = df["bar_delta"].iloc[max(0, n - 4):n - 1]
        heavy_buy = bool((deltas > cfg.absorption_delta_multiple * df["delta_abs_mean_20"].iloc[max(0, n - 4):n - 1]).any())
        absorption = (cvd2 > cvd1 or heavy_buy) and closes_below
    kind = "EXHAUSTION" if exhaustion else "ABSORPTION" if absorption else None
    return kind, n - 2 if kind else None, {
        "prior_extreme_index": t1_pos,
        "prior_cvd": cvd1,
        "candidate_cvd": cvd2,
        "candidate_bar_delta": float(current["bar_delta"]),
        "candidate_low": float(current["low"]),
        "candidate_high": float(current["high"]),
        "trigger_midpoint": float((current["high"] + current["low"]) / 2.0),
    }


def _pool_touched(df: pd.DataFrame, pool: LiquidityPool | Mapping[str, Any] | None,
                  sweep_lookback: int) -> bool:
    if pool is None:
        return True
    if isinstance(pool, LiquidityPool):
        lower, upper, level = pool.lower, pool.upper, pool.level
    else:
        level = float(pool.get("level", pool.get("price", 0.0)))
        lower = float(pool.get("lower", pool.get("low", level)))
        upper = float(pool.get("upper", pool.get("high", level)))
    tail = df.iloc[-max(1, sweep_lookback):]
    touch = bool(((tail["low"] <= upper) & (tail["high"] >= lower)).any())
    sweep = bool(((tail["low"] < lower) | (tail["high"] > upper)).any())
    return touch or sweep


def evaluate_orderflow_trigger(
    frame: pd.DataFrame,
    side: Literal["LONG", "SHORT"] | str,
    *,
    poi: LiquidityPool | Mapping[str, Any] | None = None,
    ticks: pd.DataFrame | None = None,
    config: OrderFlowConfig | None = None,
    session_allowed: bool = True,
    spread: float | None = None,
    stop_distance: float | None = None,
    tp1_distance: float | None = None,
    tp2_distance: float | None = None,
    min_net_rr_tp1: float = 1.0,
    min_net_rr_tp2: float = 1.5,
    max_spread_to_sl: float = 0.25,
) -> TriggerDecision:
    """Evaluate a setup on fully closed bars. Optional cost/RR guards fail closed."""
    cfg = config or OrderFlowConfig()
    side = str(side).upper()
    if side not in {"LONG", "SHORT"}:
        return TriggerDecision("NO_TRADE", reason="invalid_side")
    df = calculate_orderflow_features(frame, ticks=ticks, config=cfg)
    if len(df) < max(3, cfg.delta_lookback // 2):
        return TriggerDecision("NO_TRADE", reason="insufficient_bars")
    if not session_allowed:
        return TriggerDecision("NO_TRADE", reason="session_veto")
    if not _pool_touched(df, poi, cfg.sweep_lookback):
        return TriggerDecision("NO_TRADE", reason="poi_not_touched")
    div_type, absorption_pos, ctx = _divergence_context(df, side, cfg, poi)
    if div_type is None:
        return TriggerDecision("NO_TRADE", reason="no_cvd_divergence_or_absorption",
                               metrics=_last_metrics(df))
    trigger = df.iloc[-1]
    candidate = df.iloc[-2]
    avg_abs = float(trigger["delta_abs_mean_20"])
    delta = float(trigger["bar_delta"])
    if side == "LONG":
        delta_flip = delta > 0 and (
            delta >= cfg.flip_delta_multiple * avg_abs
            or float(trigger["close"]) > float(ctx["trigger_midpoint"])
        )
    else:
        delta_flip = delta < 0 and (
            delta <= -cfg.flip_delta_multiple * avg_abs
            or float(trigger["close"]) < float(ctx["trigger_midpoint"])
        )
    if not delta_flip:
        return TriggerDecision("NO_TRADE", reason="delta_flip_missing",
                               metrics={**_last_metrics(df), "cvd_divergence_type": div_type, **ctx})
    if spread is not None and stop_distance is not None:
        if stop_distance <= 0 or spread < 0 or spread / stop_distance > max_spread_to_sl:
            return TriggerDecision("NO_TRADE", reason="spread_to_sl_veto",
                                   metrics={**_last_metrics(df), "cvd_divergence_type": div_type, **ctx})
    if stop_distance is not None:
        if stop_distance <= 0:
            return TriggerDecision("NO_TRADE", reason="invalid_stop_distance")
        if tp1_distance is not None and tp1_distance / stop_distance < min_net_rr_tp1:
            return TriggerDecision("NO_TRADE", reason="tp1_rr_veto",
                                   metrics={**_last_metrics(df), "cvd_divergence_type": div_type, **ctx})
        if tp2_distance is not None and tp2_distance / stop_distance < min_net_rr_tp2:
            return TriggerDecision("NO_TRADE", reason="tp2_rr_veto",
                                   metrics={**_last_metrics(df), "cvd_divergence_type": div_type, **ctx})
    return TriggerDecision(
        "EXECUTE", side=side, reason="orderflow_confirmed",
        metrics={**_last_metrics(df), "cvd_divergence_type": div_type,
                 "absorption_bar_index": absorption_pos, **ctx},
    )


def _last_metrics(df: pd.DataFrame) -> dict[str, Any]:
    if df.empty:
        return {"bar_delta": 0.0, "cvd_value": 0.0, "delta_zscore": 0.0}
    last = df.iloc[-1]
    return {
        "bar_delta": float(last.get("bar_delta", 0.0)),
        "cvd_value": float(last.get("cvd", 0.0)),
        "delta_zscore": float(last.get("delta_zscore", 0.0)),
        "delta_source": str(last.get("delta_source", "unknown")),
    }
