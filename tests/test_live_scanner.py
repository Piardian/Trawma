import json
from pathlib import Path
import pandas as pd
import numpy as np
import pytest

from pragma.live_scanner import LiveSentinelScanner
from pragma.execution_notifier import ExecutionNotifier
from run_pipeline import main as run_pipeline_main


class DummyMT5Client:
    def __init__(self, bars):
        self._bars = bars

    def rates(self, symbol, timeframe, count=150):
        return self._bars


def make_dummy_bars(n=100):
    idx = pd.date_range("2026-01-05 08:00:00", periods=n, freq="5min", tz="UTC")
    close = 100.0 + np.sin(np.arange(n) / 5)
    return pd.DataFrame({
        "open": close,
        "high": close + 0.5,
        "low": close - 0.5,
        "close": close,
        "tick_volume": np.full(n, 1000.0),
    }, index=idx)


def test_scanner_quiet_check_prevents_reprocessing_same_bar(tmp_path):
    bars = make_dummy_bars(100)
    client = DummyMT5Client(bars)

    gate_file = tmp_path / "macro_bias_gate.json"
    gate_file.write_text(json.dumps({
        "timestamp": "2026-01-05 12:00:00",
        "capital_preservation_mode": False,
        "execution_bias_gates": {"XAUUSD": "LONG_ONLY"},
    }), encoding="utf-8")

    scanner = LiveSentinelScanner(macro_path=gate_file)
    scanner.scan_mt5_symbol(client, "XAUUSD", macro_required=True)
    key = ("XAUUSD", "M5")
    assert scanner._last_processed_bar_time.get(key) == bars.index[-1]

    # Second scan with identical bars returns immediately without processing
    res = scanner.scan_mt5_symbol(client, "XAUUSD", macro_required=True)
    assert res is None


def test_run_pipeline_cli_help(capsys):
    with pytest.raises(SystemExit) as exc:
        run_pipeline_main(["--help"])
    assert exc.value.code == 0
    captured = capsys.readouterr()
    assert "--scan-begonya" in captured.out
    assert "--scan-crypto" in captured.out
    assert "--watch" in captured.out
