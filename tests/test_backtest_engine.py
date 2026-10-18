import numpy as np
import pandas as pd
import pytest
from pragma.backtest_engine import simulate_walk_forward


def sample(n=160):
    idx=pd.date_range("2026-01-01",periods=n,freq="1min",tz="UTC")
    close=100+np.sin(np.arange(n)/5)
    return pd.DataFrame({"open":close,"high":close+0.2,"low":close-0.2,
        "close":close,"volume":np.full(n,100.0),"volume_delta":np.sin(np.arange(n))*10},index=idx)


def test_walk_forward_reports_required_metrics_and_one_position_model():
    report=simulate_walk_forward(sample(),symbol="TEST",engine="amt")
    for key in ("total_trades","win_rate","profit_factor","net_r","expectancy_r",
                "max_drawdown_r","t_stat","ci_95","funnel_counts","by_divergence_type"):
        assert key in report
    assert report["symbol"]=="TEST"
    assert report["engine"]=="amt"


def test_orderflow_strict_mode_uses_real_taker_delta():
    frame=sample()
    report=simulate_walk_forward(frame,symbol="TEST",engine="orderflow",strict_real_delta=True)
    assert report["engine"]=="orderflow"
    assert report["funnel_counts"]["strict_delta_veto"] >= 0


def test_walk_forward_rejects_insufficient_history():
    with pytest.raises(ValueError,match="100 bars"):
        simulate_walk_forward(sample(50))


def test_low_volume_day_filter_is_causal_and_handles_multiple_days():
    frame = sample(24 * 8)
    frame.index = pd.date_range("2026-01-05T00:00:00Z", periods=len(frame), freq="h")
    report = simulate_walk_forward(frame, symbol="TEST", engine="amt")
    assert report["bars_processed"] == len(frame)
    assert "low_volume_day_veto" in report["funnel_counts"]
