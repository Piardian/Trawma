import numpy as np
import pandas as pd
import pytest

from pragma.orderflow_engine import (
    OrderFlowConfig,
    calculate_orderflow_features,
    detect_orderflow_pools,
    evaluate_orderflow_trigger,
)


def bars(n=40, start="2026-01-01T00:00:00Z"):
    idx = pd.date_range(start, periods=n, freq="5min", tz="UTC")
    close = 100 + np.sin(np.arange(n) / 3) * 0.5
    return pd.DataFrame({
        "open": close - 0.05, "high": close + 0.2, "low": close - 0.2,
        "close": close, "tick_volume": np.full(n, 100.0),
    }, index=idx)


def test_bvc_delta_formula_and_flat_range_fallback():
    df = pd.DataFrame({
        "open": [9, 10, 10], "high": [11, 10, 10], "low": [9, 10, 10],
        "close": [10.5, 10, 9], "tick_volume": [100, 25, 25],
    })
    out = calculate_orderflow_features(df)
    assert out["bar_delta"].iloc[0] == pytest.approx(50.0)
    # A flat-range candle that closes below the prior close is classified as negative.
    assert out["bar_delta"].iloc[1] == pytest.approx(-25.0)
    assert out["bar_delta"].iloc[2] == pytest.approx(-25.0)
    assert out["delta_source"].iloc[0] == "bvc_synthetic_tick_volume"


def test_real_volume_delta_has_priority_over_bvc():
    df = bars(5).reset_index(names="timestamp")
    df["volume_delta"] = [10, -5, 4, -1, 2]
    result = calculate_orderflow_features(df)
    assert result["bar_delta"].tolist() == [10, -5, 4, -1, 2]
    assert (result["delta_source"] == "exchange_volume_delta").all()


def test_session_cvd_resets_at_utc_date_boundary():
    idx = pd.to_datetime(["2026-01-01T23:55:00Z", "2026-01-02T00:00:00Z", "2026-01-02T00:05:00Z"])
    df = pd.DataFrame({
        "open": [10, 10, 10], "high": [11, 11, 11], "low": [9, 9, 9],
        "close": [10.5, 9.5, 10.5], "volume_delta": [5, -2, 3],
    }, index=idx)
    assert calculate_orderflow_features(df)["cvd"].tolist() == [5, -2, 1]


def test_tick_rule_delta_aggregates_signed_ticks():
    idx = pd.date_range("2026-01-01", periods=3, freq="1min", tz="UTC")
    df = pd.DataFrame({
        "open": [100, 101, 100], "high": [101, 102, 101], "low": [99, 100, 99],
        "close": [101, 101, 100], "tick_volume": [1, 1, 1],
    }, index=idx)
    ticks = pd.DataFrame(
        {"price": [100, 101, 102, 101], "volume": [1, 2, 3, 4]},
        index=pd.to_datetime(["2026-01-01T00:00:10Z", "2026-01-01T00:00:20Z",
                              "2026-01-01T00:01:10Z", "2026-01-01T00:02:10Z"]),
    )
    result = calculate_orderflow_features(df, ticks=ticks)
    assert result["bar_delta"].tolist() == [2, 3, -4]
    assert result["delta_source"].iloc[0] == "tick_rule_estimate"


def test_detector_finds_effort_result_absorption_node():
    df = bars(40)
    df.iloc[30, df.columns.get_loc("tick_volume")] = 1000.0
    df.iloc[30, df.columns.get_loc("open")] = 100.0
    df.iloc[30, df.columns.get_loc("close")] = 100.01
    df.iloc[30, df.columns.get_loc("high")] = 100.3
    df.iloc[30, df.columns.get_loc("low")] = 99.7
    cfg = OrderFlowConfig(volume_spike_multiple=2.0, narrow_body_atr_multiple=0.25)
    assert any(p.kind == "ABSORPTION_NODE" for p in detect_orderflow_pools(df, config=cfg))


def test_trigger_fails_closed_without_divergence():
    df = bars(40).reset_index(names="timestamp")
    df["volume_delta"] = 1.0
    assert evaluate_orderflow_trigger(df, "LONG").action == "NO_TRADE"


def test_trigger_respects_session_guard():
    result = evaluate_orderflow_trigger(bars(40), "LONG", session_allowed=False)
    assert result.action == "NO_TRADE"
    assert result.reason == "session_veto"


def test_cli_modes_and_orderflow_analysis(tmp_path):
    from pragma.core import analyze_csv
    csv = tmp_path / "bars.csv"
    bars(30).reset_index(names="timestamp").to_csv(csv, index=False)
    with pytest.raises(NotImplementedError, match="AMT"):
        analyze_csv(csv, "amt")
    with pytest.raises(NotImplementedError, match="Hybrid"):
        analyze_csv(csv, "hybrid")
    result = analyze_csv(csv, "orderflow", "TEST")
    assert result["engine"] == "orderflow"
    assert result["bars"] == 30
