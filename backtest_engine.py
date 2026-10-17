"""CLI for causal walk-forward simulations on CSV or Binance Futures data."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import pandas as pd

from pragma.backtest_engine import simulate_walk_forward


def main(argv=None) -> int:
    parser=argparse.ArgumentParser(description="Trawma walk-forward trade simulator")
    parser.add_argument("--engine",choices=("amt","orderflow","hybrid"),default="amt")
    parser.add_argument("--data",type=Path,help="CSV with OHLCV and optional volume_delta")
    parser.add_argument("--symbol",default="BTCUSDT")
    parser.add_argument("--timeframe",choices=("M1","M5","M15"),default="M1")
    parser.add_argument("--count",type=int,default=50000)
    parser.add_argument("--cache-dir",type=Path,default=Path("data_cache"))
    parser.add_argument("--strict-real-delta",action="store_true")
    parser.add_argument("--fee-bps-per-side",type=float,default=4.0)
    parser.add_argument("--funding-rate",type=float,default=0.0001)
    parser.add_argument("--output",type=Path)
    args=parser.parse_args(argv)
    if args.data:
        frame=pd.read_csv(args.data)
    else:
        from pragma.data.binance_data import BinanceDataClient
        with BinanceDataClient(cache_dir=args.cache_dir) as client:
            frame=client.rates(args.symbol,args.timeframe,count=args.count)
    report=simulate_walk_forward(frame,symbol=args.symbol,engine=args.engine,
        strict_real_delta=args.strict_real_delta,fee_bps_per_side=args.fee_bps_per_side,
        funding_rate=args.funding_rate)
    rendered=json.dumps(report,indent=2,default=str,allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(rendered+"\n",encoding="utf-8")
    print(rendered)
    return 0

if __name__=="__main__":
    raise SystemExit(main())
