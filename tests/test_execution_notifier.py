import pytest
from pragma.execution_notifier import (
    SignalDeduplicator,
    OrderFlowSignal,
    format_signal_text,
    ExecutionNotifier,
)


def test_signal_deduplicator_prevents_duplicate_signals_for_same_bar():
    dedup = SignalDeduplicator(max_history=10)
    assert dedup.is_new("XAUUSD", "M5", "2026-10-18 10:00:00") is True
    assert dedup.is_new("XAUUSD", "M5", "2026-10-18 10:00:00") is False
    assert dedup.is_new("XAUUSD", "M5", "2026-10-18 10:05:00") is True
    assert dedup.is_new("BTCUSDT", "M5", "2026-10-18 10:00:00") is True


def test_format_signal_text_contains_required_metrics():
    sig = OrderFlowSignal(
        symbol="XAUUSD",
        timeframe="M5",
        side="LONG",
        entry_price=2720.50,
        stop_loss=2718.00,
        tp1=2723.00,
        tp2=2725.50,
        risk=2.50,
        bar_time="2026-10-18 10:00:00",
        poi_info={"kind": "SESSION_EXTREME", "level": 2720.0},
        delta_metrics={"bar_delta": 45.2, "cvd_value": 120.5, "delta_zscore": 2.1},
        divergence_type="ABSORPTION",
        macro_bias="LONG_ONLY",
        engine="TRAWMA_ORDERFLOW",
    )

    text = format_signal_text(sig, format_type="text")
    assert "TRAWMA ORDER FLOW" in text
    assert "XAUUSD" in text
    assert "LONG" in text
    assert "ABSORPTION" in text
    assert "2720.50000" in text

    html = format_signal_text(sig, format_type="telegram_html")
    assert "<b>XAUUSD</b>" in html or "<code>XAUUSD</code>" in html
    assert "LONG_ONLY" in html


def test_execution_notifier_dispatches_only_new_signals(capsys):
    notifier = ExecutionNotifier()
    sig = OrderFlowSignal(
        symbol="BTCUSDT",
        timeframe="M15",
        side="SHORT",
        entry_price=68000.0,
        stop_loss=68500.0,
        tp1=67500.0,
        tp2=67000.0,
        risk=500.0,
        bar_time="2026-10-18 11:00:00",
        poi_info={"kind": "SWING_HIGH", "level": 68050.0},
        delta_metrics={"bar_delta": -12.5, "cvd_value": -80.0, "delta_zscore": -1.8},
        divergence_type="ABSORPTION",
        macro_bias="SHORT_ONLY",
    )

    first = notifier.notify(sig)
    captured = capsys.readouterr()
    assert first is True
    assert "TRAWMA ORDER FLOW CANLI SINYAL" in captured.out

    second = notifier.notify(sig)
    assert second is False
