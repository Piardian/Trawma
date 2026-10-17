"""Session VWAP and volume-weighted standard deviation bands."""
from __future__ import annotations

import numpy as np
import pandas as pd


def calculate_session_vwap(
    bars: pd.DataFrame,
    session_key: str | None = None,
    band_multipliers: tuple[float, ...] = (1.0, 1.5, 2.0, 2.5),
) -> pd.DataFrame:
    """Return cumulative session VWAP, weighted sigma, and deviation bands."""
    required = {"high", "low", "close", "tick_volume"}
    missing = required.difference(bars.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    if bars.empty:
        raise ValueError("bars cannot be empty")
    if not all(np.isfinite(bars[c].to_numpy(dtype=float)).all() for c in required):
        raise ValueError("bars contain NaN or infinite values")
    if (bars["tick_volume"].astype(float) < 0).any():
        raise ValueError("tick_volume cannot be negative")
    if any((not np.isfinite(k) or k < 0) for k in band_multipliers):
        raise ValueError("band multipliers must be finite and non-negative")

    out = bars.copy()
    price = (out["high"].astype(float) + out["low"].astype(float) + out["close"].astype(float)) / 3.0
    volume = out["tick_volume"].astype(float)
    if session_key is None:
        groups = pd.Series(0, index=out.index)
    else:
        if session_key not in out.columns:
            raise ValueError(f"session_key column {session_key!r} not found")
        if out[session_key].isna().any():
            raise ValueError("session grouping column contains null values")
        groups = out[session_key]

    cumulative_volume = volume.groupby(groups, sort=False).cumsum()
    cumulative_pv = (price * volume).groupby(groups, sort=False).cumsum()
    cumulative_p2v = (price.pow(2) * volume).groupby(groups, sort=False).cumsum()
    valid = cumulative_volume > 0
    vwap = pd.Series(np.nan, index=out.index, dtype=float)
    second_moment = pd.Series(np.nan, index=out.index, dtype=float)
    vwap.loc[valid] = cumulative_pv.loc[valid] / cumulative_volume.loc[valid]
    second_moment.loc[valid] = cumulative_p2v.loc[valid] / cumulative_volume.loc[valid]
    variance = (second_moment - vwap.pow(2)).clip(lower=0.0)
    sigma = np.sqrt(variance)
    out["typical_price"] = price
    out["cumulative_volume"] = cumulative_volume
    out["vwap"] = vwap
    out["vwap_sigma"] = sigma
    for mult in sorted(set(band_multipliers)):
        suffix = str(mult).replace(".", "p")
        out[f"vwap_upper_{suffix}sigma"] = vwap + mult * sigma
        out[f"vwap_lower_{suffix}sigma"] = vwap - mult * sigma
    return out


class VWAPEngine:
    def __init__(self, band_multipliers: tuple[float, ...] = (1.0, 1.5, 2.0, 2.5)):
        self.band_multipliers = band_multipliers

    def calculate(self, bars: pd.DataFrame, session_key: str | None = None) -> pd.DataFrame:
        return calculate_session_vwap(bars, session_key=session_key, band_multipliers=self.band_multipliers)
