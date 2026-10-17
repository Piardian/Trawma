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
    assert out["delta_source"].iloc[0] == "synthetic_bvc"


def test_real_volume_delta_has_priority_over_bvc():
    df = bars(5).reset_index(names="timestamp")
    df["volume_delta"] = [10, -5, 4, -1, 2]
    result = calculate_orderflow_features(df)
    assert result["bar_delta"].tolist() == [10, -5, 4, -1, 2]
    assert (result["delta_source"] == "real_taker").all()


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
    assert result["delta_source"].iloc[0] == "tick_rule"


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



def _divergence_fixture(side, kind):
    # 26 bars: a confirmed prior extreme, a penultimate divergence/absorption
    # candle, and a final closed delta-flip confirmation candle.
    n = 26
    idx = pd.date_range("2026-01-01T00:00:00Z", periods=n, freq="1min")
    close = np.full(n, 100.0)
    high = np.full(n, 100.4)
    low = np.full(n, 99.6)
    delta = np.zeros(n)
    # prior window local extreme at bar 18
    if side == "LONG":
        low[18] = 98.0
        close[18] = 98.4
        delta[18] = -10.0
        if kind == "EXHAUSTION":
            low[24] = 97.8
            close[24] = 98.2
            delta[24] = -2.0
        else:
            low[24] = 98.2
            close[24] = 98.5
            delta[24] = -30.0
        delta[25] = 20.0
        close[25] = 99.0
    else:
        high[18] = 102.0
        close[18] = 101.6
        delta[18] = 10.0
        if kind == "EXHAUSTION":
            high[24] = 102.2
            close[24] = 101.8
            delta[24] = 2.0
        else:
            high[24] = 101.8
            close[24] = 101.5
            delta[24] = 30.0
        delta[25] = -20.0
        close[25] = 101.0
    open_ = close.copy()
    open_[25] = close[25] - (0.2 if side == "LONG" else -0.2)
    volume = np.full(n, 100.0)
    frame = pd.DataFrame({
        "open": open_, "high": high, "low": low, "close": close,
        "volume_delta": delta, "tick_volume": volume,
    }, index=idx)
    # Keep earlier CVD values below/above the designated pivot in a controlled way.
    if kind == "EXHAUSTION":
        if side == "LONG":
            delta[19:24] = 4.0
            delta[24] = -2.0
        else:
            delta[19:24] = -4.0
            delta[24] = 2.0
    else:
        if side == "LONG":
            delta[19:24] = 0.0
            delta[24] = -30.0
        else:
            delta[19:24] = 0.0
            delta[24] = 30.0
    frame["volume_delta"] = delta
    return frame


@pytest.mark.parametrize(("side", "kind"), [
    ("LONG", "EXHAUSTION"), ("LONG", "ABSORPTION"),
    ("SHORT", "EXHAUSTION"), ("SHORT", "ABSORPTION"),
])
def test_controlled_divergence_fixtures(side, kind):
    frame = _divergence_fixture(side, kind)
    result = evaluate_orderflow_trigger(frame, side)
    assert result.action == "EXECUTE", result
    assert result.side == side
    assert result.metrics["cvd_divergence_type"] == kind


def test_strict_real_delta_rejects_synthetic_bvc():
    frame = bars(40)
    result = evaluate_orderflow_trigger(frame, "LONG", strict_real_delta=True)
    assert result.action == "NO_TRADE"
    assert result.reason == "strict_real_delta_required"


def test_delta_quality_and_source_are_explicit():
    real = bars(5)
    real["volume_delta"] = [1, -1, 2, -2, 3]
    real_out = calculate_orderflow_features(real)
    synthetic_out = calculate_orderflow_features(bars(5))
    assert set(real_out["delta_source"]) == {"real_taker"}
    assert set(real_out["delta_quality"]) == {"REAL"}
    assert set(synthetic_out["delta_source"]) == {"synthetic_bvc"}
    assert set(synthetic_out["delta_quality"]) == {"ESTIMATED"}
