"""CLI entrypoint for Trawma pipeline and live sentinel watch mode."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

from pragma.core import run_cli
from pragma.live_scanner import run_sentinel_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Trawma Pipeline and Live Sentinel Watcher")
    
    # Sentinel live mode options
    sentinel_group = parser.add_argument_group("Sentinel Live Watcher Options")
    sentinel_group.add_argument("--scan-begonya", action="store_true", help="Scan MT5 Whitelist instruments with Begonya macro gate")
    sentinel_group.add_argument("--scan-crypto", action="store_true", help="Scan Binance Futures crypto instruments (BTC, SOL)")
    sentinel_group.add_argument("--watch", action="store_true", help="Run continuous 24/7 watch loop")
    sentinel_group.add_argument("--interval-sec", type=int, default=15, help="Scan loop sleep interval in seconds")
    sentinel_group.add_argument("--refresh-macro", action="store_true", help="Refresh macro gate on each iteration")
    sentinel_group.add_argument("--macro-path", type=Path, default=None, help="Custom path to macro_bias_gate.json")
    sentinel_group.add_argument("--execute-mode", choices=("notify_only", "paper", "mt5_live"), default="notify_only", help="Execution mode")
    sentinel_group.add_argument("--confirm-live", action="store_true", help="Explicit confirmation required to send real orders in mt5_live mode")
    sentinel_group.add_argument("--risk-pct", type=float, default=0.5, help="Risk percentage per trade (default: 0.5%%)")

    # Offline / single-run options (preserved from core.py)
    core_group = parser.add_argument_group("Offline Feature/POI Analysis Options")
    core_group.add_argument("--engine", choices=("amt", "orderflow", "hybrid"), default="orderflow")
    core_group.add_argument("--source", choices=("binance", "mt5", "csv"), default="csv")
    core_group.add_argument("--data", type=Path)
    core_group.add_argument("--symbol", default="BTCUSDT")
    core_group.add_argument("--timeframe", choices=("M1", "M5", "M15", "M30", "H1", "H4", "D1"), default="M1")
    core_group.add_argument("--count", type=int, default=50000)
    core_group.add_argument("--tick-count", type=int, default=200000)
    core_group.add_argument("--strict-real-delta", action="store_true")
    core_group.add_argument("--output", type=Path)

    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.scan_begonya or args.scan_crypto or args.watch:
        return run_sentinel_pipeline(
            scan_begonya=args.scan_begonya,
            scan_crypto=args.scan_crypto,
            watch=args.watch,
            interval_sec=args.interval_sec,
            refresh_macro=args.refresh_macro,
            macro_path=args.macro_path,
            execute_mode=args.execute_mode,
            confirm_live=args.confirm_live,
            risk_pct=args.risk_pct,
        )

    # Otherwise forward to classic offline analysis using the parsed args
    return run_cli("Trawma pipeline", argv=argv)


if __name__ == "__main__":
    raise SystemExit(main())
