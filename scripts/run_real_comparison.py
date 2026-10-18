"""Run selective absorption-only comparisons on 100k Binance Futures M1 bars.

M5/M15 are causally aggregated from closed M1 bars, preserving taker delta.
Altcoin entries are gated by BTC's contemporaneous price/CVD regime.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import pandas as pd
from pragma.data.binance_data import BinanceDataClient
from pragma.backtest_engine import simulate_walk_forward

def aggregate(frame: pd.DataFrame, minutes: int) -> pd.DataFrame:
    if not isinstance(frame.index,pd.DatetimeIndex):
        raise ValueError("Binance adapter must return a DatetimeIndex")
    rules=f"{minutes}min"
    agg={"open":"first","high":"max","low":"min","close":"last"}
    for c in ("volume","tick_volume","buy_volume","sell_volume","volume_delta",
              "quote_asset_volume","taker_buy_base_asset_volume","taker_buy_quote_asset_volume"):
        if c in frame.columns: agg[c]="sum"
    grouped=frame.resample(rules,label="left",closed="left")
    result=grouped.agg(agg).dropna(subset=["open","high","low","close"])
    # Reject partial M5/M15 candles at the tail and any bucket with missing M1 bars.
    complete=grouped["open"].count()==minutes
    return result.loc[complete.reindex(result.index,fill_value=False)]

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--count",type=int,default=100000)
    p.add_argument("--symbols",nargs="+",default=["BTCUSDT","SOLUSDT"])
    p.add_argument("--cache-dir",default="data_cache")
    p.add_argument("--output",default="artifacts/real_walk_forward_comparison.json")
    a=p.parse_args()
    if a.count<100000: raise SystemExit("--count must be >= 100000 for the requested benchmark")
    reports=[]
    with BinanceDataClient(cache_dir=a.cache_dir, futures_only=True) as client:
        btc_m1=client.rates("BTCUSDT","M1",count=a.count)
        if len(btc_m1)<a.count:
            raise RuntimeError(f"BTCUSDT: requested {a.count} closed M1 bars, received {len(btc_m1)}")
        for symbol in a.symbols:
            m1=btc_m1 if symbol.upper()=="BTCUSDT" else client.rates(symbol,"M1",count=a.count)
            if len(m1)<a.count:
                raise RuntimeError(f"{symbol}: requested {a.count} closed M1 bars, received {len(m1)}; refusing partial benchmark")
            if "volume_delta" not in m1 or not m1["volume_delta"].notna().all():
                raise RuntimeError(f"{symbol}: missing Binance taker-derived volume_delta")
            source="real_taker"
            btc_by_tf={1:btc_m1,5:aggregate(btc_m1,5),15:aggregate(btc_m1,15)}
            for minutes,frame in [(1,m1),(5,aggregate(m1,5)),(15,aggregate(m1,15))]:
                frame=frame.copy()
                if "tick_volume" not in frame and "volume" in frame: frame["tick_volume"]=frame["volume"]
                btc_frame=None if symbol.upper()=="BTCUSDT" else btc_by_tf[minutes]
                for engine in ("orderflow","hybrid"):
                    strict=True
                    report=simulate_walk_forward(frame,symbol=symbol,engine=engine,strict_real_delta=strict,btc_frame=btc_frame)
                    report.update({"timeframe":f"M{minutes}","bars_downloaded_m1":len(m1),"btc_leader_bars":len(btc_by_tf[minutes]) if btc_frame is not None else None,
                                   "delta_source":source if minutes==1 else "real_taker_aggregated"})
                    reports.append(report)
    out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(reports,indent=2,allow_nan=False,default=str)+"\n",encoding="utf-8")
    summary=[{k:r[k] for k in ("symbol","timeframe","engine","bars_processed","total_trades","win_rate","profit_factor","net_r","expectancy_r","max_drawdown_r","t_stat","ci_95","delta_source") if k in r} for r in reports]
    print(json.dumps(summary,indent=2,allow_nan=False,default=str))
    print(f"Full report: {out}")

if __name__=="__main__": main()
