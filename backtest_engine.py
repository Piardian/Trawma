"""CLI for causal walk-forward simulations on Binance, MT5, or CSV data."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import pandas as pd

from pragma.backtest_engine import simulate_walk_forward


def _load_data(args):
    if args.source == "csv":
        if not args.data:
            raise ValueError("--source csv requires --data PATH")
        frame = pd.read_csv(args.data)
        if "timestamp" in frame.columns:
            frame.index = pd.to_datetime(frame.pop("timestamp"), utc=True)
        elif "time" in frame.columns:
            frame.index = pd.to_datetime(frame.pop("time"), utc=True)
        return frame, None, None
    if args.source == "binance":
        from pragma.data.binance_data import BinanceDataClient, filter_closed_bars
        with BinanceDataClient(cache_dir=args.cache_dir, futures_only=True) as client:
            frame = client.rates(args.symbol, args.timeframe, count=args.count)
        frame = filter_closed_bars(frame, args.timeframe)
        btc = None
        if args.symbol.upper() not in {"BTCUSDT", "BTCUSD", "BTC"}:
            with BinanceDataClient(cache_dir=args.cache_dir, futures_only=True) as client:
                btc = client.rates("BTCUSDT", args.timeframe, count=args.count)
            btc = filter_closed_bars(btc, args.timeframe)
        return frame, None, btc
    from pragma.data.mt5_data import MT5DataClient, filter_closed_bars
    from datetime import timedelta
    with MT5DataClient() as client:
        frame = filter_closed_bars(client.rates(args.symbol, args.timeframe, count=args.count), args.timeframe)
        ticks = None
        if args.tick_count > 0 and not frame.empty:
            start = frame.index[0].to_pydatetime()
            ticks = client.ticks_from(args.symbol, start, count=args.tick_count)
    return frame, ticks, None


def main(argv=None) -> int:
    parser=argparse.ArgumentParser(description="Trawma walk-forward trade simulator")
    parser.add_argument("--engine",choices=("amt","orderflow","hybrid"),default="orderflow")
    parser.add_argument("--source",choices=("binance","mt5","csv"),default=None)
    parser.add_argument("--data",type=Path,help="CSV path; required when --source csv")
    parser.add_argument("--symbol",default="BTCUSDT")
    parser.add_argument("--timeframe",choices=("M1","M5","M15","M30","H1","H4","D1"),default="M1")
    parser.add_argument("--count",type=int,default=50000)
    parser.add_argument("--tick-count",type=int,default=200000,help="MT5 ticks fetched for Tick Rule delta; set 0 to disable")
    parser.add_argument("--cache-dir",type=Path,default=Path("data_cache"))
    parser.add_argument("--strict-real-delta",action="store_true")
    parser.add_argument("--fee-bps-per-side",type=float,default=4.0)
    parser.add_argument("--funding-rate",type=float,default=0.0001)
    parser.add_argument("--output",type=Path)
    args=parser.parse_args(argv)
    args.source = args.source or ("csv" if args.data else "binance")
    frame, ticks, btc = _load_data(args)
    report=simulate_walk_forward(frame,symbol=args.symbol,engine=args.engine,
        strict_real_delta=args.strict_real_delta,fee_bps_per_side=args.fee_bps_per_side,
        funding_rate=args.funding_rate,ticks=ticks,btc_frame=btc)
    rendered=json.dumps(report,indent=2,default=str,allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(rendered+"\n",encoding="utf-8")
    print(rendered)
    return 0

if __name__=="__main__":
    raise SystemExit(main())
