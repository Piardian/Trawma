"""Auction Market Theory profiles from MT5 bars or tick observations.

MT5 M1 tick_volume is activity count, not centralized traded volume. For bar
profiles this engine spreads each bar's tick_volume equally across every price
bin intersecting [low, high], conserving each bar's volume. It remains an
approximation: OHLC bars do not reveal the true intrabar volume-at-price path.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class VolumeProfile:
    histogram: pd.DataFrame
    poc: float
    vah: float
    val: float
    total_volume: float
    value_area_fraction: float
    lvn_levels: tuple[float, ...]


@dataclass(frozen=True)
class PreviousSessionProfile:
    """Completed prior session levels; pd* names are stable interface fields."""
    session: object
    pdPOC: float
    pdVAH: float
    pdVAL: float
    pdLVN: tuple[float, ...]
    profile: VolumeProfile


@dataclass(frozen=True)
class DevelopingProfile:
    """Profile calculated only from the active session data supplied to it."""
    session: object
    profile: VolumeProfile


@dataclass(frozen=True)
class SessionProfiles:
    previous: PreviousSessionProfile
    developing: DevelopingProfile


def _validate_bars(bars: pd.DataFrame) -> pd.DataFrame:
    required = {"high", "low", "close", "tick_volume"}
    missing = required.difference(bars.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    if bars.empty:
        raise ValueError("bars cannot be empty")
    frame = bars.loc[:, ["high", "low", "close", "tick_volume"]].astype(float).copy()
    if not np.isfinite(frame.to_numpy()).all():
        raise ValueError("bars contain NaN or infinite values")
    if (frame["high"] < frame["low"]).any():
        raise ValueError("high must be greater than or equal to low")
    if (frame["tick_volume"] < 0).any():
        raise ValueError("tick_volume cannot be negative")
    if frame["tick_volume"].sum() <= 0:
        raise ValueError("total tick volume must be positive")
    return frame


def _build_profile(
    volumes: np.ndarray,
    edges: np.ndarray,
    value_area_fraction: float,
    lvn_percentile: float,
) -> VolumeProfile:
    bins = len(volumes)
    centers = (edges[:-1] + edges[1:]) / 2.0
    total = float(volumes.sum())
    if total <= 0:
        raise ValueError("No positive volume fell inside the profile range")

    poc_idx = int(np.argmax(volumes))
    target = total * value_area_fraction
    left = right = poc_idx
    accumulated = float(volumes[poc_idx])
    while accumulated < target and (left > 0 or right < bins - 1):
        left_vol = float(volumes[left - 1]) if left > 0 else -np.inf
        right_vol = float(volumes[right + 1]) if right < bins - 1 else -np.inf
        if left_vol >= right_vol:
            left -= 1
            accumulated += float(volumes[left])
        else:
            right += 1
            accumulated += float(volumes[right])

    threshold = float(np.percentile(volumes, lvn_percentile))
    local_min = ((volumes[1:-1] <= volumes[:-2])
                 & (volumes[1:-1] <= volumes[2:])
                 & (volumes[1:-1] <= threshold))
    lvn_indices = np.flatnonzero(local_min) + 1
    hist = pd.DataFrame({
        "price_low": edges[:-1],
        "price_high": edges[1:],
        "price": centers,
        "tick_volume": volumes.astype(float),
        "is_value_area": (np.arange(bins) >= left) & (np.arange(bins) <= right),
        "is_lvn": np.isin(np.arange(bins), lvn_indices),
    })
    return VolumeProfile(
        histogram=hist,
        poc=float(centers[poc_idx]),
        vah=float(edges[right + 1]),
        val=float(edges[left]),
        total_volume=total,
        value_area_fraction=float(accumulated / total),
        lvn_levels=tuple(float(centers[i]) for i in lvn_indices),
    )


def calculate_volume_profile(
    bars: pd.DataFrame,
    bins: int = 100,
    value_area_fraction: float = 0.70,
    lvn_percentile: float = 20.0,
) -> VolumeProfile:
    """Build a bar-range-uniform profile and derive POC, VAH/VAL, and LVNs.

    Each bar's tick volume is divided equally among all bins intersecting its
    [low, high] range. A zero-range bar is assigned wholly to the containing
    bin. The sum of histogram volume equals the sum of input bar volume
    (subject only to floating-point rounding).

    Value Area expands from POC toward the larger adjacent-volume bin until
    the configured fraction is covered. This is not the same as one sigma.
    """
    if not isinstance(bins, int) or bins < 2:
        raise ValueError("bins must be an integer >= 2")
    if not 0 < value_area_fraction <= 1:
        raise ValueError("value_area_fraction must be in (0, 1]")
    if not 0 <= lvn_percentile <= 100:
        raise ValueError("lvn_percentile must be in [0, 100]")

    frame = _validate_bars(bars)
    low_edge = float(frame["low"].min())
    high_edge = float(frame["high"].max())
    if np.isclose(low_edge, high_edge):
        pad = max(abs(low_edge) * 1e-9, np.finfo(float).eps * 16)
        low_edge -= pad
        high_edge += pad
    edges = np.linspace(low_edge, high_edge, bins + 1)
    bar_lows = frame["low"].to_numpy()
    bar_highs = frame["high"].to_numpy()
    bar_volumes = frame["tick_volume"].to_numpy(dtype=float)

    # Rows are bars, columns are bins. Include any bin intersecting [low, high].
    intersects = ((bar_lows[:, None] < edges[1:][None, :])
                  & (bar_highs[:, None] > edges[:-1][None, :]))
    zero_range = np.isclose(bar_lows, bar_highs)
    if zero_range.any():
        intersects[zero_range, :] = False
        idx = np.searchsorted(edges, bar_lows[zero_range], side="right") - 1
        idx = np.clip(idx, 0, bins - 1)
        intersects[np.flatnonzero(zero_range), idx] = True

    counts = intersects.sum(axis=1)
    if (counts == 0).any():
        raise ValueError("Could not assign one or more bars to price bins")
    per_bin_volume = bar_volumes / counts
    histogram_volumes = (intersects * per_bin_volume[:, None]).sum(axis=0)
    return _build_profile(histogram_volumes, edges, value_area_fraction, lvn_percentile)


def calculate_session_profiles(
    bars: pd.DataFrame,
    session_key: str,
    as_of=None,
    bins: int = 100,
    value_area_fraction: float = 0.70,
    lvn_percentile: float = 20.0,
) -> SessionProfiles:
    """Return completed previous-session levels and the active developing profile.

    bars must include a session label column. If as_of is provided, rows later
    than that timestamp are discarded before any calculation. For backtests,
    call this at each decision timestamp or pass as_of. Supplying a full future
    day without an as_of cutoff would itself introduce look-ahead.
    The latest remaining session label is active; the preceding distinct label
    is the completed previous session.
    """
    if session_key not in bars.columns:
        raise ValueError(f"session_key column {session_key!r} not found")
    if bars.empty:
        raise ValueError("bars cannot be empty")
    frame = bars.copy().sort_index()
    if as_of is not None:
        cutoff = pd.Timestamp(as_of)
        if isinstance(frame.index, pd.DatetimeIndex):
            if frame.index.tz is not None and cutoff.tzinfo is None:
                cutoff = cutoff.tz_localize(frame.index.tz)
            elif frame.index.tz is None and cutoff.tzinfo is not None:
                cutoff = cutoff.tz_localize(None)
        frame = frame.loc[frame.index <= cutoff]
    if frame.empty:
        raise ValueError("No bars exist at or before as_of")
    if frame[session_key].isna().any():
        raise ValueError("session labels cannot be null")

    labels = frame[session_key].drop_duplicates().tolist()
    if len(labels) < 2:
        raise ValueError("At least two sessions are required")
    active_label, previous_label = labels[-1], labels[-2]
    engine_args = dict(bins=bins, value_area_fraction=value_area_fraction,
                       lvn_percentile=lvn_percentile)
    previous_profile = calculate_volume_profile(
        frame.loc[frame[session_key] == previous_label], **engine_args
    )
    active_profile = calculate_volume_profile(
        frame.loc[frame[session_key] == active_label], **engine_args
    )
    return SessionProfiles(
        previous=PreviousSessionProfile(
            session=previous_label, pdPOC=previous_profile.poc,
            pdVAH=previous_profile.vah, pdVAL=previous_profile.val,
            pdLVN=previous_profile.lvn_levels, profile=previous_profile,
        ),
        developing=DevelopingProfile(session=active_label, profile=active_profile),
    )


class AMTEngine:
    """Configurable AMT profile and session-profile calculator."""
    def __init__(self, bins: int = 100, value_area_fraction: float = 0.70,
                 lvn_percentile: float = 20.0):
        self.bins = bins
        self.value_area_fraction = value_area_fraction
        self.lvn_percentile = lvn_percentile

    def calculate(self, bars: pd.DataFrame) -> VolumeProfile:
        return calculate_volume_profile(
            bars, bins=self.bins, value_area_fraction=self.value_area_fraction,
            lvn_percentile=self.lvn_percentile,
        )

    def calculate_sessions(self, bars: pd.DataFrame, session_key: str,
                           as_of=None) -> SessionProfiles:
        return calculate_session_profiles(
            bars, session_key=session_key, as_of=as_of, bins=self.bins,
            value_area_fraction=self.value_area_fraction,
            lvn_percentile=self.lvn_percentile,
        )
