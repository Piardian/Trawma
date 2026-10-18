import json
from datetime import datetime, timezone
import pytest
from pragma.macro_gate import (
    parse_begonya_timestamp,
    read_macro_bias,
    is_macro_aligned,
    find_macro_bias_file,
)


def test_parse_begonya_naive_timestamp_uses_utc_plus_3():
    dt = parse_begonya_timestamp("2026-10-18 12:57:47")
    assert dt is not None
    # 12:57:47 in UTC+3 is 09:57:47 in UTC
    assert dt.tzinfo == timezone.utc
    assert dt.hour == 9
    assert dt.minute == 57
    assert dt.second == 47


def test_macro_bias_capital_preservation_mode(tmp_path):
    gate_file = tmp_path / "macro_bias_gate.json"
    gate_file.write_text(json.dumps({
        "timestamp": "2026-10-18 12:00:00",
        "capital_preservation_mode": True,
        "execution_bias_gates": {"XAUUSD": "LONG_ONLY"},
    }), encoding="utf-8")

    decision = read_macro_bias("XAUUSD", path=gate_file)
    assert not decision.is_executable
    assert decision.direction == 0
    assert decision.capital_preservation_mode is True
    assert not is_macro_aligned("XAUUSD", "LONG", path=gate_file)


def test_macro_bias_event_freeze(tmp_path):
    gate_file = tmp_path / "macro_bias_gate.json"
    gate_file.write_text(json.dumps({
        "timestamp": "2026-10-18 12:00:00",
        "capital_preservation_mode": False,
        "execution_bias_gates": {"XAUUSD": "EVENT_FREEZE"},
    }), encoding="utf-8")

    decision = read_macro_bias("XAUUSD", path=gate_file)
    assert not decision.is_executable
    assert decision.direction == 0
    assert "FREEZE" in decision.raw_gate


def test_macro_bias_directional_alignment_and_aliases(tmp_path):
    gate_file = tmp_path / "macro_bias_gate.json"
    gate_file.write_text(json.dumps({
        "timestamp": "2026-10-18 12:00:00",
        "capital_preservation_mode": False,
        "execution_bias_gates": {
            "XAUUSD": "LONG_ONLY",
            "BTC": "LONG_ONLY",
            "EURUSD": "SHORT_ONLY",
            "SOL": "NEUTRAL_RANGE",
        },
    }), encoding="utf-8")

    # XAUUSD LONG_ONLY
    xau = read_macro_bias("XAUUSD", path=gate_file)
    assert xau.is_executable
    assert xau.direction == 1
    assert xau.allowed_side == "LONG"
    assert is_macro_aligned("XAUUSD", "LONG", path=gate_file) is True
    assert is_macro_aligned("XAUUSD", "SHORT", path=gate_file) is False

    # BTCUSDT alias to BTC
    btc = read_macro_bias("BTCUSDT", path=gate_file)
    assert btc.direction == 1
    assert btc.allowed_side == "LONG"

    # EURUSD SHORT_ONLY
    eur = read_macro_bias("EURUSD", path=gate_file)
    assert eur.direction == -1
    assert eur.allowed_side == "SHORT"
    assert is_macro_aligned("EURUSD", "SHORT", path=gate_file) is True
    assert is_macro_aligned("EURUSD", "LONG", path=gate_file) is False

    # SOL NEUTRAL_RANGE
    sol = read_macro_bias("SOLUSDT", path=gate_file)
    assert not sol.is_executable
    assert sol.direction == 0
