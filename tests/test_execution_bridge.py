from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import pytest
import pandas as pd

from pragma.execution_bridge import (
    calculate_mt5_lot_size,
    PaperOrderTracker,
    ExecutionBridge,
)
from pragma.execution_notifier import OrderFlowSignal


@dataclass
class DummySymbolInfo:
    trade_tick_size: float = 0.01
    trade_tick_value: float = 1.0  # $1 per 0.01 move for 1 lot
    volume_min: float = 0.01
    volume_step: float = 0.01
    volume_max: float = 50.0
    point: float = 0.01


def test_calculate_mt5_lot_size():
    info = DummySymbolInfo()
    balance = 10000.0  # $10,000 balance
    risk_pct = 0.5     # $50 risk (0.5%)
    entry = 2700.0
    sl = 2695.0        # $5 distance = 500 ticks of 0.01 -> $500 loss per 1 lot

    lot = calculate_mt5_lot_size(balance, entry, sl, info, risk_pct=risk_pct)
    # Expected lot = $50 / $500 = 0.10 lot
    assert lot == 0.10


def test_calculate_mt5_lot_size_clamps_to_min_and_max():
    info = DummySymbolInfo(volume_min=0.05, volume_max=10.0)
    # Extremely small risk that would be 0.001 lot -> clamped to 0.05
    lot_min = calculate_mt5_lot_size(100.0, 2700.0, 2690.0, info, risk_pct=0.1)
    assert lot_min == 0.05

    # Extremely huge risk -> clamped to 10.0
    lot_max = calculate_mt5_lot_size(1000000.0, 2700.0, 2699.9, info, risk_pct=5.0)
    assert lot_max == 10.0


def test_execution_bridge_requires_confirm_live():
    with pytest.raises(PermissionError, match="--confirm-live"):
        ExecutionBridge(mode="mt5_live", confirm_live=False)

    bridge_notify = ExecutionBridge(mode="notify_only")
    assert bridge_notify.mode == "notify_only"


def test_paper_order_tracker_lifecycle(tmp_path):
    paper_file = tmp_path / "paper_positions.json"
    tracker = PaperOrderTracker(storage_path=paper_file)

    sig = OrderFlowSignal(
        symbol="XAUUSD",
        timeframe="M5",
        side="LONG",
        entry_price=2700.0,
        stop_loss=2695.0,
        tp1=2705.0,
        tp2=2710.0,
        risk=5.0,
        bar_time="2026-01-05 10:00:00",
        poi_info={},
        delta_metrics={},
    )

    pos = tracker.open_position(sig, lot=0.1)
    assert pos["status"] == "OPEN"
    assert pos["bars_held"] == 0

    # Bar 1: Price touches TP1 (2705.0) -> partial take profit & breakeven
    bar_tp1 = {"symbol": "XAUUSD", "low": 2698.0, "high": 2706.0, "close": 2704.0}
    t1 = pd.Timestamp("2026-01-05 10:05:00", tz="UTC")
    tracker.update(bar_tp1, t1)

    updated_pos = tracker.open_positions[0]
    assert updated_pos["tp1_taken"] is True
    assert updated_pos["stop_loss"] == 2700.0  # Moved to breakeven!
    assert updated_pos["qty_remaining"] == 0.5
    assert updated_pos["realized_r"] == 0.5

    # Bar 2: Price reaches TP2 (2710.0) -> full exit
    bar_tp2 = {"symbol": "XAUUSD", "low": 2704.0, "high": 2711.0, "close": 2710.5}
    t2 = pd.Timestamp("2026-01-05 10:10:00", tz="UTC")
    tracker.update(bar_tp2, t2)

    assert len(tracker.open_positions) == 0
    closed_pos = tracker._positions[0]
    assert closed_pos["status"] == "CLOSED"
    assert closed_pos["exit_reason"] == "TP2"
    assert closed_pos["realized_r"] == 1.5  # 0.5R from TP1 + 0.5*2R from TP2 = 1.5R


def test_paper_order_tracker_time_stop_after_16_bars(tmp_path):
    paper_file = tmp_path / "paper_positions.json"
    tracker = PaperOrderTracker(storage_path=paper_file)

    sig = OrderFlowSignal(
        symbol="XAUUSD",
        timeframe="M5",
        side="LONG",
        entry_price=2700.0,
        stop_loss=2695.0,
        tp1=2705.0,
        tp2=2710.0,
        risk=5.0,
        bar_time="2026-01-05 10:00:00",
        poi_info={},
        delta_metrics={},
    )
    tracker.open_position(sig, lot=0.1)

    # Simulate 16 sideways bars without hitting SL or TP1
    stale_bar = {"symbol": "XAUUSD", "low": 2698.0, "high": 2702.0, "close": 2701.0}
    times = pd.date_range("2026-01-05 10:05:00", periods=16, freq="5min", tz="UTC")
    for t in times:
        tracker.update(stale_bar, t)

    assert len(tracker.open_positions) == 0
    closed = tracker._positions[0]
    assert closed["status"] == "CLOSED"
    assert closed["exit_reason"] == "TIME_STOP"


def test_paper_order_tracker_monthly_stats_and_scoreboard(tmp_path):
    paper_file = tmp_path / "paper_positions.json"
    tracker = PaperOrderTracker(storage_path=paper_file)

    # 1. Add a winning trade (TP2 -> +1.5R)
    tracker._positions.append({
        "id": "p1",
        "symbol": "SOLUSDT",
        "status": "CLOSED",
        "tp1_taken": True,
        "exit_reason": "TP2",
        "entry_time": "2026-10-01 10:00:00",
        "exit_time": "2026-10-01 11:00:00",
        "realized_r": 1.5,
    })

    # 2. Add a loss (SL -> -1.0R)
    tracker._positions.append({
        "id": "p2",
        "symbol": "XAUUSD",
        "status": "CLOSED",
        "tp1_taken": False,
        "exit_reason": "SL",
        "entry_time": "2026-10-02 14:00:00",
        "exit_time": "2026-10-02 14:15:00",
        "realized_r": -1.0,
    })

    # 3. Add a breakeven trade (TP1 + BE SL -> +0.5R)
    tracker._positions.append({
        "id": "p3",
        "symbol": "SOLUSDT",
        "status": "CLOSED",
        "tp1_taken": True,
        "exit_reason": "SL",
        "entry_time": "2026-10-03 09:00:00",
        "exit_time": "2026-10-03 10:30:00",
        "realized_r": 0.5,
    })

    # 4. Add a time-stop trade (TIME_STOP -> -0.1R)
    tracker._positions.append({
        "id": "p4",
        "symbol": "XAUUSD",
        "status": "CLOSED",
        "tp1_taken": False,
        "exit_reason": "TIME_STOP",
        "entry_time": "2026-10-04 12:00:00",
        "exit_time": "2026-10-04 13:20:00",
        "realized_r": -0.1,
    })

    stats = tracker.get_monthly_stats(target_month="2026-10")
    assert stats["total_closed"] == 4
    assert stats["tp_count"] == 2  # p1 and p3 (realized > 0 with tp1 taken)
    assert stats["sl_count"] == 1  # p2
    assert stats["be_count"] == 1  # p3
    assert stats["time_stop_count"] == 1  # p4
    assert pytest.approx(stats["net_r"], 0.01) == 0.90  # 1.5 - 1.0 + 0.5 - 0.1 = 0.90R
    assert stats["win_rate"] == 50.0  # 2 wins out of 4

    scoreboard = tracker.format_scoreboard_html(target_month="2026-10")
    assert "AYLIK PERFORMANS SAYACI" in scoreboard
    assert "+0.90R" in scoreboard
    assert "TP (Kâr Al)  :</b> 2" in scoreboard
    assert "SL (Zarar Durdur):</b> 1" in scoreboard
    assert "BE (Başabaş) :</b> 1" in scoreboard

