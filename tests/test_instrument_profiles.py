import pytest
from pragma.instrument_profiles import (
    WHITELIST_INSTRUMENTS,
    BLACKLIST_FOREX_SYMBOLS,
    is_whitelisted,
    is_blacklisted,
    get_profile,
)


def test_whitelist_contains_proven_instruments():
    assert "BTCUSDT" in WHITELIST_INSTRUMENTS
    assert "SOLUSDT" in WHITELIST_INSTRUMENTS
    assert "XAUUSD" in WHITELIST_INSTRUMENTS

    btc = WHITELIST_INSTRUMENTS["BTCUSDT"]
    assert btc.optimal_timeframe == "M15"
    assert btc.source == "binance"
    assert btc.strict_real_delta is True

    sol = WHITELIST_INSTRUMENTS["SOLUSDT"]
    assert sol.optimal_timeframe == "M15"
    assert sol.lead_symbol == "BTCUSDT"

    xau = WHITELIST_INSTRUMENTS["XAUUSD"]
    assert xau.optimal_timeframe == "M5"
    assert xau.source == "mt5"


def test_blacklist_contains_negative_ev_forex_symbols():
    expected = {"EURUSD", "CADCHF", "USDJPY", "CADJPY", "EURGBP", "GBPJPY"}
    assert expected.issubset(BLACKLIST_FOREX_SYMBOLS)

    for sym in expected:
        assert is_blacklisted(sym)
        assert not is_whitelisted(sym)


def test_symbol_lookup_normalisation():
    assert is_whitelisted("btc/usdt")
    assert is_whitelisted("xau_usd")
    assert is_blacklisted("eur-usd")
    assert is_blacklisted("cad_jpy")

    profile = get_profile("BTC-USDT")
    assert profile is not None
    assert profile.symbol == "BTCUSDT"
