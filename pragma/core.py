"""Small first-phase CLI helpers for Trawma.

This is deliberately a research scaffold, not an execution/live-trading program. The AMT
implementation and calibrated stop/target construction are not present yet.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from .orderflow_engine import (
    EngineMode,
    OrderFlowConfig,
    calculate_orderflow_features,
    detect_orderflow_pools,
)


def add_engine_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--engine", choices=("amt", "orderflow", "hybrid"), default="amt",
        help="Research engine to use (default: amt).",
    )
    parser.add_argument("--data", type=Path, help="CSV market data file.")
    parser.add_argument("--symbol", default="UNKNOWN", help="Instrument label for the report.")
    parser.add_argument("--output", type=Path, help="Optional JSON report output path.")


def analyze_csv(path: Path, engine: EngineMode, symbol: str = "UNKNOWN") -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    data = pd.read_csv(path)
    if engine == "amt":
        raise NotImplementedError(
            "The independent Trawma AMT engine has not been implemented yet; "
            "use --engine orderflow for Phase 1."
        )
    if engine == "hybrid":
        raise NotImplementedError(
            "Hybrid needs an AMT POI provider. Implement AMT first; hybrid will not silently "
            "substitute orderflow POIs."
        )
    features = calculate_orderflow_features(data, config=OrderFlowConfig())
    pools = detect_orderflow_pools(data, config=OrderFlowConfig())
    last = features.iloc[-1] if len(features) else None
    return {
        "symbol": symbol,
        "engine": engine,
        "bars": int(len(data)),
        "delta_source": str(last["delta_source"]) if last is not None else "empty",
        "last_bar_delta": float(last["bar_delta"]) if last is not None else None,
        "last_cvd": float(last["cvd"]) if last is not None else None,
        "detected_pool_count": len(pools),
        "pools": [
            {"kind": p.kind, "side": p.side, "level": p.level, "lower": p.lower,
             "upper": p.upper, "timestamp": str(p.timestamp), "source": p.source,
             "strength": p.strength}
            for p in pools
        ],
        "note": "Feature/pool report only; this command does not claim a trade backtest.",
    }


def run_cli(description: str, argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=description)
    add_engine_argument(parser)
    args = parser.parse_args(argv)
    if args.data is None:
        parser.error("--data PATH is required for analysis")
    report = analyze_csv(args.data, args.engine, args.symbol)
    formatted = json.dumps(report, indent=2, default=str)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(formatted + "\n", encoding="utf-8")
    print(formatted)
    return 0
