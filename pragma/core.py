"""CSV analysis and CLI dispatch for Trawma."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Any
import pandas as pd
from .orderflow_engine import EngineMode, OrderFlowConfig, calculate_orderflow_features, detect_orderflow_pools


def analyze_csv(path: Path, engine: EngineMode, symbol: str = "UNKNOWN") -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    if engine == "amt":
        raise NotImplementedError("Trawma AMT engine has not been implemented yet.")
    if engine == "hybrid":
        raise NotImplementedError("Hybrid requires the Trawma AMT POI provider; no silent fallback.")
    data = pd.read_csv(path)
    cfg = OrderFlowConfig()
    features = calculate_orderflow_features(data, config=cfg)
    pools = detect_orderflow_pools(data, config=cfg)
    last = features.iloc[-1] if len(features) else None
    return {
        "symbol": symbol, "engine": engine, "bars": len(data),
        "delta_source": str(last["delta_source"]) if last is not None else "empty",
        "last_bar_delta": float(last["bar_delta"]) if last is not None else None,
        "last_cvd": float(last["cvd"]) if last is not None else None,
        "detected_pool_count": len(pools),
        "pools": [{"kind": p.kind, "side": p.side, "level": p.level, "lower": p.lower,
                   "upper": p.upper, "timestamp": str(p.timestamp), "source": p.source}
                  for p in pools],
        "note": "Feature/pool report only; not a trade-level backtest.",
    }


def run_cli(description: str, argv=None) -> int:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--engine", choices=("amt", "orderflow", "hybrid"), default="amt")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--symbol", default="UNKNOWN")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    report = analyze_csv(args.data, args.engine, args.symbol)
    output = json.dumps(report, indent=2, default=str)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + "\n", encoding="utf-8")
    print(output)
    return 0
