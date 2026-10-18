"""CSV pipeline analysis for Trawma's selectable engines."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Any
import pandas as pd
from .orderflow_engine import EngineMode, OrderFlowConfig, calculate_orderflow_features, detect_orderflow_pools


def analyze_csv(path: Path, engine: EngineMode, symbol: str = "UNKNOWN", strict_real_delta: bool = False, ticks: pd.DataFrame | None = None) -> dict[str, Any]:
    path=Path(path)
    if not path.exists(): raise FileNotFoundError(path)
    data=pd.read_csv(path)
    if data.empty: raise ValueError("input data is empty")
    cfg=OrderFlowConfig(strict_real_delta=strict_real_delta)
    features=calculate_orderflow_features(data,ticks=ticks,config=cfg)
    last=features.iloc[-1]
    result={
      "symbol":symbol,"engine":engine,"bars":len(data),
      "delta_source":str(last["delta_source"]),"delta_quality":str(last["delta_quality"]),
      "last_bar_delta":float(last["bar_delta"]),"last_cvd":float(last["cvd"]),
      "last_delta_zscore":float(last["delta_zscore"]),
    }
    if engine in ("orderflow","hybrid"):
      pools=detect_orderflow_pools(data,config=cfg)
      result["pools"]=[{"kind":p.kind,"side":p.side,"level":p.level,"lower":p.lower,"upper":p.upper,
                        "timestamp":str(p.timestamp),"source":p.source} for p in pools]
      result["detected_pool_count"]=len(pools)
    if engine in ("amt","hybrid"):
      from .amt_engine import calculate_volume_profile
      from .vwap_engine import calculate_session_vwap
      bars=data.copy()
      if "tick_volume" not in bars:
        if "volume" in bars: bars["tick_volume"]=bars["volume"]
        else: raise ValueError("AMT/VWAP requires tick_volume or volume")
      profile=calculate_volume_profile(bars[["high","low","close","tick_volume"]].tail(min(500,len(bars))),bins=50)
      vwap=calculate_session_vwap(bars[["high","low","close","tick_volume"]]).iloc[-1]
      result["amt"]={"pdPOC":profile.poc,"pdVAH":profile.vah,"pdVAL":profile.val,"pdLVN":list(profile.lvn_levels)}
      result["vwap"]={"vwap":float(vwap["vwap"]),"sigma":float(vwap["vwap_sigma"]),
        "bands":{k:float(vwap[k]) for k in vwap.index if "upper_" in k or "lower_" in k}}
    result["note"]="Feature/POI report only; use backtest_engine.py for trade simulation."
    return result


def run_cli(description: str, argv=None) -> int:
    parser=argparse.ArgumentParser(description=description)
    parser.add_argument("--engine",choices=("amt","orderflow","hybrid"),default="orderflow")
    parser.add_argument("--source",choices=("binance","mt5","csv"),default="csv")
    parser.add_argument("--data",type=Path)
    parser.add_argument("--symbol",default="BTCUSDT")
    parser.add_argument("--timeframe",choices=("M1","M5","M15","M30","H1","H4","D1"),default="M1")
    parser.add_argument("--count",type=int,default=50000)
    parser.add_argument("--tick-count",type=int,default=200000)
    parser.add_argument("--strict-real-delta",action="store_true")
    parser.add_argument("--output",type=Path)
    args=parser.parse_args(argv)
    ticks=None
    if args.source=="csv":
      if not args.data: parser.error("--source csv requires --data PATH")
      report=analyze_csv(args.data,args.engine,args.symbol,args.strict_real_delta)
    else:
      import tempfile
      if args.source=="binance":
        from .data.binance_data import BinanceDataClient, filter_closed_bars
        with BinanceDataClient() as client: data=client.rates(args.symbol,args.timeframe,count=args.count)
        data=filter_closed_bars(data,args.timeframe)
      else:
        from .data.mt5_data import MT5DataClient, filter_closed_bars
        with MT5DataClient() as client:
          data=filter_closed_bars(client.rates(args.symbol,args.timeframe,count=args.count),args.timeframe)
          if args.tick_count>0 and not data.empty:
            ticks=client.ticks_from(args.symbol,data.index[0].to_pydatetime(),count=args.tick_count)
      if data.empty: raise ValueError("No closed bars returned by selected data source")
      with tempfile.NamedTemporaryFile(suffix=".csv",mode="w",encoding="utf-8",delete=False) as handle:
        temp_path=Path(handle.name)
        data.rename_axis("timestamp").reset_index().to_csv(handle,index=False)
      try: report=analyze_csv(temp_path,args.engine,args.symbol,args.strict_real_delta,ticks=ticks)
      finally: temp_path.unlink(missing_ok=True)
    report["source"]=args.source
    output=json.dumps(report,indent=2,default=str,allow_nan=False)
    if args.output:
      args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(output+"\n",encoding="utf-8")
    print(output)
    return 0
