"""Causal bar-by-bar trade simulator for Trawma.

Signal decisions use only closed bars through t; entries are filled at t+1 open.
OHLC collisions resolve conservatively (SL before TP). Costs are charged on both
entry and exit; funding is charged once per 8 hours while a position is open.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
import math
from typing import Any
import numpy as np
import pandas as pd

from .orderflow_engine import (
    OrderFlowConfig, calculate_orderflow_features, detect_orderflow_pools,
    evaluate_orderflow_trigger,
)

@dataclass
class Position:
    side: str
    entry: float
    stop: float
    tp1: float
    tp2: float
    risk: float
    entry_index: int
    divergence: str
    qty_remaining: float = 1.0
    tp1_taken: bool = False
    realized_r: float = 0.0
    funding_events: int = 0

def _metrics(trades: list[dict], funnel: dict, symbol: str, engine: str) -> dict[str, Any]:
    rs=np.array([t["net_r"] for t in trades],dtype=float)
    wins=rs[rs>0]; losses=rs[rs<0]
    pf=float(wins.sum()/abs(losses.sum())) if len(losses) and abs(losses.sum()) else (float("inf") if len(wins) else 0.0)
    equity=np.cumsum(rs)
    peak=np.maximum.accumulate(np.r_[0.0,equity])[1:] if len(equity) else np.array([])
    dd=peak-equity if len(equity) else np.array([])
    sd=float(rs.std(ddof=1)) if len(rs)>1 else 0.0
    tstat=float(rs.mean()/(sd/math.sqrt(len(rs)))) if len(rs)>1 and sd>0 else 0.0
    se=sd/math.sqrt(len(rs)) if len(rs)>1 else 0.0
    return {
      "symbol":symbol,"engine":engine,"total_trades":len(trades),
      "win_rate":float((rs>0).mean()) if len(rs) else 0.0,"profit_factor":pf,
      "net_r":float(rs.sum()) if len(rs) else 0.0,
      "expectancy_r":float(rs.mean()) if len(rs) else 0.0,
      "max_drawdown_r":float(dd.max()) if len(dd) else 0.0,"t_stat":tstat,
      "ci_95":[float(rs.mean()-1.96*se),float(rs.mean()+1.96*se)] if len(rs) else [0.0,0.0],
      "funnel_counts":funnel,
      "by_divergence_type":{
        k:{"trades":sum(t["divergence_type"]==k for t in trades),
           "win_rate":float(np.mean([t["net_r"]>0 for t in trades if t["divergence_type"]==k]))
             if any(t["divergence_type"]==k for t in trades) else 0.0}
        for k in ("ABSORPTION","EXHAUSTION")
      },
      "trades":trades,
    }

def simulate_walk_forward(frame: pd.DataFrame, *, symbol: str="UNKNOWN",
                          engine: str="orderflow", strict_real_delta: bool=False,
                          fee_bps_per_side: float=4.0, funding_rate: float=0.0001,
                          atr_multiple: float=1.0, rr_tp1: float=1.0, rr_tp2: float=2.0,
                          config: OrderFlowConfig|None=None) -> dict[str,Any]:
    if engine not in {"amt","orderflow","hybrid"}: raise ValueError("engine must be amt, orderflow, or hybrid")
    if len(frame)<100: raise ValueError("at least 100 bars required for walk-forward simulation")
    cfg=config or OrderFlowConfig(strict_real_delta=strict_real_delta)
    df=frame.copy()
    ren={str(c).lower().replace(" ","_"):c for c in df.columns}
    for k in ("open","high","low","close"):
        if k not in ren: raise ValueError(f"missing {k}")
    df=df.rename(columns={ren[k]:k for k in ("open","high","low","close")})
    if "timestamp" in df.columns: df.index=pd.to_datetime(df.pop("timestamp"),utc=True)
    elif not isinstance(df.index,pd.DatetimeIndex): raise ValueError("frame requires timestamp column or DatetimeIndex")
    df=df.sort_index()
    if "volume_delta" not in df and "tick_volume" not in df and "volume" not in df:
        raise ValueError("need real volume_delta or volume/tick_volume")
    features=calculate_orderflow_features(df,config=cfg)
    funnel={"bars_seen":0,"poi_touch":0,"divergence":0,"delta_flip":0,"cost_veto":0,"entries":0,"stop_exits":0,"tp1":0,"tp2":0,"end_of_data":0,"strict_delta_veto":0}
    trades=[]; pos=None
    fee=max(0.0,fee_bps_per_side)/10000.0
    for i in range(50,len(df)-1):
        bar=df.iloc[i]; nextbar=df.iloc[i+1]; ts=df.index[i]
        funnel["bars_seen"]+=1
        # Existing position management first; stop wins if SL and TP share a bar.
        if pos is not None:
            held_seconds=(ts-df.index[pos.entry_index]).total_seconds() if hasattr(ts-df.index[pos.entry_index],"total_seconds") else (i-pos.entry_index)*60
            while held_seconds >= (pos.funding_events+1)*8*3600:
                pos.realized_r -= funding_rate*pos.entry*pos.qty_remaining/pos.risk
                pos.funding_events += 1
            lo,hi=float(bar.low),float(bar.high)
            stop_hit=(lo<=pos.stop if pos.side=="LONG" else hi>=pos.stop)
            if stop_hit:
                exit_price=pos.stop
                gross=((exit_price-pos.entry) if pos.side=="LONG" else (pos.entry-exit_price))/pos.risk
                pos.realized_r += pos.qty_remaining*gross - fee*(pos.entry+exit_price)*pos.qty_remaining/pos.risk
                trades.append({"entry_time":str(df.index[pos.entry_index]),"exit_time":str(ts),"side":pos.side,"net_r":pos.realized_r,"exit_reason":"SL","tp1_taken":pos.tp1_taken,"divergence_type":pos.divergence})
                funnel["stop_exits"]+=1; pos=None; continue
            tp1_hit=(hi>=pos.tp1 if pos.side=="LONG" else lo<=pos.tp1)
            if not pos.tp1_taken and tp1_hit:
                gross=((pos.tp1-pos.entry) if pos.side=="LONG" else (pos.entry-pos.tp1))/pos.risk
                pos.realized_r += 0.5*gross - fee*(pos.entry+pos.tp1)*0.5/pos.risk
                pos.qty_remaining=0.5; pos.tp1_taken=True; pos.stop=pos.entry; funnel["tp1"]+=1
            tp2_hit=(hi>=pos.tp2 if pos.side=="LONG" else lo<=pos.tp2)
            if pos.tp1_taken and tp2_hit:
                gross=((pos.tp2-pos.entry) if pos.side=="LONG" else (pos.entry-pos.tp2))/pos.risk
                pos.realized_r += pos.qty_remaining*gross - fee*(pos.entry+pos.tp2)*pos.qty_remaining/pos.risk
                trades.append({"entry_time":str(df.index[pos.entry_index]),"exit_time":str(ts),"side":pos.side,"net_r":pos.realized_r,"exit_reason":"TP2","tp1_taken":True,"divergence_type":pos.divergence})
                funnel["tp2"]+=1; pos=None
            if pos is not None: continue
        # Prefix-only signal calculation prevents look-ahead.
        prefix=df.iloc[:i+1]
        atr=float(features.atr_14.iloc[i])
        if not np.isfinite(atr) or atr<=0: continue
        side=None; decision=None; div="UNKNOWN"
        if engine in ("orderflow","hybrid"):
            poi=None
            if engine=="orderflow":
                pools=detect_orderflow_pools(prefix,config=cfg)
                touched=[p for p in pools if float(bar.low)<=p.upper and float(bar.high)>=p.lower]
                if touched:
                    funnel["poi_touch"]+=1
                    pool=touched[-1]; side=pool.side if pool.side in ("LONG","SHORT") else None; poi=pool
            else:
                # Hybrid POIs are AMT/VWAP reference levels calculated on the observed prefix only.
                from .amt_engine import calculate_volume_profile
                from .vwap_engine import calculate_session_vwap
                prof_bars=prefix.copy()
                if "tick_volume" not in prof_bars:
                    prof_bars["tick_volume"]=prof_bars["volume"] if "volume" in prof_bars else 0.0
                try:
                    vp=calculate_volume_profile(prof_bars[["high","low","close","tick_volume"]].tail(min(500,len(prof_bars))),bins=50)
                    vwap=calculate_session_vwap(prof_bars[["high","low","close","tick_volume"]]).iloc[-1]
                    candidates=[("LONG",vp.val),("SHORT",vp.vah),("LONG",float(vwap["vwap"])),("SHORT",float(vwap["vwap"]))]
                    hit=[(s,l) for s,l in candidates if abs(float(bar.close)-l)<=0.35*atr]
                    if hit: side,level=hit[-1]; poi={"level":level,"lower":level-0.25*atr,"upper":level+0.25*atr}
                except (ValueError,KeyError,TypeError): side=None
            if side:
                decision=evaluate_orderflow_trigger(prefix,side,poi=poi,config=cfg,strict_real_delta=strict_real_delta)
                if decision.action=="EXECUTE":
                    div=str(decision.metrics.get("cvd_divergence_type") or "UNKNOWN"); funnel["divergence"]+=1; funnel["delta_flip"]+=1
                else:
                    if decision.reason=="strict_real_delta_required": funnel["strict_delta_veto"]+=1
                    side=None
        else:
            # AMT-only control: causal z-score excursion from rolling typical-price mean.
            typical=(prefix.high+prefix.low+prefix.close)/3.0
            mean=typical.rolling(50,min_periods=20).mean().iloc[-1]
            std=typical.rolling(50,min_periods=20).std(ddof=0).iloc[-1]
            z=(float(bar.close)-float(mean))/float(std) if pd.notna(std) and std>0 else 0.0
            if z<=-2.0: side="LONG"
            elif z>=2.0: side="SHORT"
        if side not in ("LONG","SHORT"): continue
        entry=float(nextbar.open)
        risk=max(atr*atr_multiple,entry*fee*2.5)
        stop=entry-risk if side=="LONG" else entry+risk
        tp1=entry+rr_tp1*risk*(1 if side=="LONG" else -1)
        tp2=entry+rr_tp2*risk*(1 if side=="LONG" else -1)
        # Net-cost guard: spread/fee cost must not consume too much of the initial risk.
        if (2*fee*entry)/risk>0.25: funnel["cost_veto"]+=1; continue
        pos=Position(side,entry,stop,tp1,tp2,risk,i+1,div)
        funnel["entries"]+=1
    if pos is not None:
        last=df.iloc[-1]; ts=df.index[-1]; exit_price=float(last.close)
        gross=((exit_price-pos.entry) if pos.side=="LONG" else (pos.entry-exit_price))/pos.risk
        pos.realized_r += pos.qty_remaining*gross - fee*(pos.entry+exit_price)*pos.qty_remaining/pos.risk
        trades.append({"entry_time":str(df.index[pos.entry_index]),"exit_time":str(ts),"side":pos.side,"net_r":pos.realized_r,"exit_reason":"END_OF_DATA","tp1_taken":pos.tp1_taken,"divergence_type":pos.divergence})
        funnel["end_of_data"]+=1
    report=_metrics(trades,funnel,symbol,engine)
    report["bars_processed"]=len(df)
    report["delta_source"]=str(features["delta_source"].iloc[-1]) if len(features) else "empty"
    return report
