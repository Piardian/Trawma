"""Instrument profiles, whitelists, blacklists and optimal timeframe mappings for Trawma.

Based on empirical walk-forward testing:
- BTCUSDT (M15, Binance Futures, Order Flow Absorption): PF 1.02, Net R +0.23R, WR 58.3%
- SOLUSDT (M15, Binance Futures, BTC CVD Gated): PF 3.06, Net R +2.25R, WR 80.0%
- XAUUSD (M5, MT5 Terminal, Order Flow Absorption): PF 1.20, Net R +1.24R, WR 61.5%
- Unanchored MT5 FX pairs (EURUSD, CADCHF, USDJPY, CADJPY, EURGBP, GBPJPY): Negative EV, Blacklisted.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class InstrumentProfile:
    symbol: str
    asset_class: str  # "crypto" | "commodity" | "forex"
    source: str       # "binance" | "mt5"
    optimal_timeframe: str  # "M5" | "M15"
    engine: str = "orderflow"
    strict_real_delta: bool = False
    lead_symbol: str | None = None
    fee_bps_per_side: float = 4.0
    funding_rate: float = 0.0001
    proven_profit_factor: float = 1.0
    proven_net_r: float = 0.0
    notes: str = ""


# Empirically proven optimal instruments (Phase 6 Walk-Forward Verified)
WHITELIST_INSTRUMENTS: dict[str, InstrumentProfile] = {
    "BTCUSDT": InstrumentProfile(
        symbol="BTCUSDT",
        asset_class="crypto",
        source="binance",
        optimal_timeframe="M15",
        engine="orderflow_absorption",
        strict_real_delta=True,
        lead_symbol=None,
        fee_bps_per_side=4.0,
        funding_rate=0.0001,
        proven_profit_factor=1.18,
        proven_net_r=0.45,
        notes="Binance 100k M15 Orderflow Absorption (OOS PF 1.18, WR 60%)",
    ),
    "LINKUSDT": InstrumentProfile(
        symbol="LINKUSDT",
        asset_class="crypto",
        source="binance",
        optimal_timeframe="M15",
        engine="cvd_momentum",
        strict_real_delta=True,
        lead_symbol="BTCUSDT",
        fee_bps_per_side=4.0,
        funding_rate=0.0001,
        proven_profit_factor=2.45,
        proven_net_r=5.09,
        notes="Binance 100k M15 CVD Momentum Pullback (OOS PF 2.45, WR 58.3%, Net +5.09R)",
    ),
    "NAS100": InstrumentProfile(
        symbol="NAS100",
        asset_class="index",
        source="mt5",
        optimal_timeframe="M15",
        engine="hybrid_absorption",
        strict_real_delta=False,
        lead_symbol=None,
        fee_bps_per_side=2.5,
        funding_rate=0.0,
        proven_profit_factor=1.73,
        proven_net_r=5.06,
        notes="MT5 50k M15 Begonya Macro Hybrid Absorption (OOS PF 1.73, WR 66.7%, Net +5.06R)",
    ),
    "SOLUSDT": InstrumentProfile(
        symbol="SOLUSDT",
        asset_class="crypto",
        source="binance",
        optimal_timeframe="M15",
        engine="orderflow_absorption",
        strict_real_delta=True,
        lead_symbol="BTCUSDT",
        fee_bps_per_side=4.0,
        funding_rate=0.0001,
        proven_profit_factor=3.06,
        proven_net_r=2.25,
        notes="Binance 100k M15 BTC CVD leadership gated",
    ),
    "XAUUSD": InstrumentProfile(
        symbol="XAUUSD",
        asset_class="commodity",
        source="mt5",
        optimal_timeframe="M5",
        engine="orderflow_absorption",
        strict_real_delta=False,
        lead_symbol=None,
        fee_bps_per_side=1.0,
        funding_rate=0.0,
        proven_profit_factor=1.20,
        proven_net_r=1.24,
        notes="MT5 50k M5 Orderflow Absorption-Only",
    ),
}

# Negative EV Forex pairs from 50k benchmarks
BLACKLIST_FOREX_SYMBOLS: frozenset[str] = frozenset({
    "EURUSD",
    "CADCHF",
    "USDJPY",
    "CADJPY",
    "EURGBP",
    "GBPJPY",
})

# Unanchored MT5 indices without macro bias
BLACKLIST_UNANCHORED_INDICES: frozenset[str] = frozenset({
    "US100",
    "SP500",
    "US500",
})

BLACKLIST_SYMBOLS: frozenset[str] = BLACKLIST_FOREX_SYMBOLS.union(BLACKLIST_UNANCHORED_INDICES)


def is_whitelisted(symbol: str) -> bool:
    norm = symbol.upper().replace("/", "").replace("-", "").replace("_", "")
    return norm in WHITELIST_INSTRUMENTS


def is_blacklisted(symbol: str) -> bool:
    norm = symbol.upper().replace("/", "").replace("-", "").replace("_", "")
    return norm in BLACKLIST_SYMBOLS


def get_profile(symbol: str) -> InstrumentProfile | None:
    norm = symbol.upper().replace("/", "").replace("-", "").replace("_", "")
    return WHITELIST_INSTRUMENTS.get(norm)
