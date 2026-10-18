"""Trawma Order Flow core: sourced delta, session CVD, pools, and trigger checks."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Literal, Mapping
import numpy as np
import pandas as pd

EngineMode = Literal["amt", "orderflow", "hybrid"]

@dataclass(frozen=True)
class OrderFlowConfig:
    session_timezone: str = "UTC"
    session_reset: Literal["daily", "none"] = "daily"
    delta_lookback: int = 20
    atr_period: int = 14
    volume_sma_period: int = 20
    swing_left: int = 10
    swing_right: int = 10
    pool_lookback: int = 100
    volume_spike_multiple: float = 2.5
    narrow_body_atr_multiple: float = 0.25
    poi_atr_multiplier: float = 0.25
    divergence_lookback: int = 15
    sweep_lookback: int = 3
    absorption_delta_multiple: float = 1.5
    flip_delta_multiple: float = 1.0
    allow_exhaustion: bool = False
    active_hours_utc: tuple[int, int] = (8, 22)
    filter_low_volume_days: bool = True
    require_delta_source: bool = True
    strict_real_delta: bool = False

@dataclass(frozen=True)
class LiquidityPool:
    kind: str
    side: Literal["LONG", "SHORT", "BOTH"]
    level: float
    lower: float
    upper: float
    timestamp: Any
    source: str
    strength: float = 0.0
    metadata: Mapping[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class TriggerDecision:
    action: Literal["EXECUTE", "NO_TRADE"]
    side: Literal["LONG", "SHORT", "NONE"] = "NONE"
    reason: str = ""
    metrics: Mapping[str, Any] = field(default_factory=dict)
    @property
    def execute(self) -> bool:
        return self.action == "EXECUTE"

def _normalise(frame: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame")
    df = frame.copy()
    names = {}
    valid = {"open","high","low","close","volume","tick_volume","volume_delta","timestamp","time","session","price","tick_price","last","bid","ask"}
    for c in df.columns:
        k = str(c).strip().lower().replace(" ","_")
        if k in valid: names[c] = k
    df = df.rename(columns=names)
    if "timestamp" in df.columns:
        df.index = pd.to_datetime(df.pop("timestamp"), utc=True, errors="coerce")
    elif "time" in df.columns and not isinstance(df.index,pd.DatetimeIndex):
        df.index = pd.to_datetime(df.pop("time"), utc=True, errors="coerce")
    if not isinstance(df.index,pd.DatetimeIndex):
        df.index = pd.RangeIndex(len(df), name=df.index.name or "bar")
    elif df.index.tz is None: df.index = df.index.tz_localize("UTC")
    else: df.index = df.index.tz_convert("UTC")
    missing = {"open","high","low","close"} - set(df.columns)
    if missing: raise ValueError(f"Missing OHLC columns: {sorted(missing)}")
    for col in ("open","high","low","close","volume","tick_volume","volume_delta","price","tick_price"):
        if col in df: df[col] = pd.to_numeric(df[col],errors="coerce")
    return df

def _tr(df):
    prev = df.close.shift(1)
    return pd.concat([(df.high-df.low).abs(),(df.high-prev).abs(),(df.low-prev).abs()],axis=1).max(axis=1)

def calculate_atr(frame: pd.DataFrame, period: int = 14) -> pd.Series:
    return _tr(_normalise(frame)).rolling(max(1,int(period)),min_periods=1).mean().rename("atr_14")

def _tick_rule(ticks: pd.DataFrame, bars: pd.DataFrame) -> pd.Series:
    t = ticks.copy()
    cols={}
    for c in t.columns:
        k=str(c).strip().lower().replace(" ","_")
        if k in {"timestamp","time","price","tick_price","last","volume","tick_volume","volume_real"}: cols[c]=k
    t=t.rename(columns=cols)
    if "timestamp" in t.columns: t.index=pd.to_datetime(t.pop("timestamp"),utc=True,errors="coerce")
    elif "time" in t.columns and not isinstance(t.index,pd.DatetimeIndex): t.index=pd.to_datetime(t.pop("time"),utc=True,errors="coerce")
    if not isinstance(t.index,pd.DatetimeIndex): raise ValueError("ticks need timestamp or DatetimeIndex")
    t.index=t.index.tz_localize("UTC") if t.index.tz is None else t.index.tz_convert("UTC")
    pc="price" if "price" in t else "tick_price" if "tick_price" in t else "last" if "last" in t else None
    if pc is None: raise ValueError("ticks need price or tick_price")
    t=t.sort_index()
    move=pd.to_numeric(t[pc],errors="coerce").diff()
    sign=np.sign(move).replace(0,np.nan).ffill().fillna(0.0)
    vc="volume_real" if "volume_real" in t else "volume" if "volume" in t else "tick_volume" if "tick_volume" in t else None
    vol=pd.to_numeric(t[vc],errors="coerce").fillna(1.0) if vc else pd.Series(1.0,index=t.index)
    positions=bars.index.get_indexer(t.index,method="pad")
    valid=positions>=0
    sums=np.zeros(len(bars),dtype=float)
    np.add.at(sums,positions[valid],(sign*vol).to_numpy(dtype=float)[valid])
    return pd.Series(sums,index=bars.index,name="bar_delta")

def calculate_orderflow_features(frame: pd.DataFrame, *, ticks: pd.DataFrame|None=None, config: OrderFlowConfig|None=None) -> pd.DataFrame:
    cfg=config or OrderFlowConfig()
    df=_normalise(frame)
    if df.empty:
        for c in ("bar_delta","cvd","delta_abs_mean_20","delta_zscore","atr_14","vol_sma_20","delta_source","delta_quality"): df[c]=pd.Series(dtype="object" if c=="delta_source" else float,index=df.index)
        return df
    if "volume_delta" in df and df.volume_delta.notna().any():
        delta=df.volume_delta.fillna(0.0).astype(float); source="real_taker"
    elif ticks is not None:
        delta=_tick_rule(ticks,df); source="tick_rule"
    else:
        vc="tick_volume" if "tick_volume" in df else "volume" if "volume" in df else None
        if vc is None:
            if cfg.require_delta_source: raise ValueError("Need volume_delta, tick data, tick_volume or volume")
            delta=pd.Series(0.0,index=df.index); source="unavailable_zero"
        else:
            vol=df[vc].fillna(0).clip(lower=0)
            spread=df.high-df.low
            bvc=vol*(2*(df.close-df.low)/spread.replace(0,np.nan)-1)
            sign=np.where(df.close>=df.close.shift(1),1.0,-1.0)
            sign[0]=1.0
            delta=bvc.where(spread>0,vol*sign).replace([np.inf,-np.inf],np.nan).fillna(0.0)
            source="synthetic_bvc"
    df["bar_delta"]=delta.astype(float)
    if "session" in df:
        sessions=df.session.astype("string").fillna("UNKNOWN")
    elif isinstance(df.index,pd.DatetimeIndex):
        sessions=pd.Series(df.index.tz_convert(cfg.session_timezone).date,index=df.index)
    else: sessions=pd.Series(0,index=df.index)
    df["cvd"]=df.bar_delta.groupby(sessions,sort=False).cumsum() if cfg.session_reset=="daily" else df.bar_delta.cumsum()
    n=max(1,cfg.delta_lookback)
    df["delta_abs_mean_20"]=df.bar_delta.abs().rolling(n,min_periods=1).mean()
    mean=df.bar_delta.rolling(n,min_periods=1).mean()
    std=df.bar_delta.rolling(n,min_periods=2).std(ddof=0).replace(0,np.nan)
    df["delta_zscore"]=((df.bar_delta-mean)/std).fillna(0.0)
    df["atr_14"]=_tr(df).rolling(max(1,cfg.atr_period),min_periods=1).mean()
    vc="tick_volume" if "tick_volume" in df else "volume" if "volume" in df else None
    df["vol_sma_20"]=df[vc].rolling(max(1,cfg.volume_sma_period),min_periods=1).mean() if vc else np.nan
    df["delta_source"]=source
    df["delta_quality"]="REAL_TAKER" if source=="real_taker" else "TICK_ESTIMATE" if source=="tick_rule" else "ESTIMATED"
    return df

def detect_orderflow_pools(frame: pd.DataFrame, *, config: OrderFlowConfig|None=None, profile: Any|None=None, previous_session_levels: Mapping[str,float]|None=None) -> list[LiquidityPool]:
    cfg=config or OrderFlowConfig()
    df=calculate_orderflow_features(frame,config=cfg)
    if df.empty: return []
    mult=float(getattr(profile,"poi_atr_multiplier",cfg.poi_atr_multiplier))
    pools=[]
    left,right=max(1,cfg.swing_left),max(1,cfg.swing_right)
    start=max(left,len(df)-cfg.pool_lookback); end=len(df)-right
    for i in range(start,max(start,end)):
        lo,hi=float(df.low.iloc[i]),float(df.high.iloc[i])
        if len(df.low.iloc[i-left:i])!=left or len(df.low.iloc[i+1:i+right+1])!=right: continue
        atr=float(df.atr_14.iloc[i]); w=max(0,mult*atr)
        if lo<=df.low.iloc[i-left:i].min() and lo<df.low.iloc[i+1:i+right+1].min():
            pools.append(LiquidityPool("SWING_LOW","LONG",lo,lo-w,lo+w,df.index[i],"confirmed_swing",metadata={"pivot_index":i}))
        if hi>=df.high.iloc[i-left:i].max() and hi>df.high.iloc[i+1:i+right+1].max():
            pools.append(LiquidityPool("SWING_HIGH","SHORT",hi,hi-w,hi+w,df.index[i],"confirmed_swing",metadata={"pivot_index":i}))
    levels=dict(previous_session_levels or {})
    if not levels and isinstance(df.index,pd.DatetimeIndex) and len(df)>1:
        keys=pd.Series(df.index.date,index=df.index)
        days=list(pd.unique(keys))
        if len(days)>1:
            prev=df.loc[keys==days[-2]]
            if not prev.empty: levels={"Session High":float(prev.high.max()),"Session Low":float(prev.low.min())}
    for name,val in levels.items():
        val=float(val)
        if not np.isfinite(val): continue
        side="LONG" if any(x in name.lower() for x in ("low","val")) else "SHORT" if any(x in name.lower() for x in ("high","vah")) else "BOTH"
        atr=float(df.atr_14.iloc[-1]); w=max(0,mult*atr)
        pools.append(LiquidityPool("SESSION_EXTREME",side,val,val-w,val+w,df.index[-1],"previous_session",metadata={"level_name":name}))
    vc="tick_volume" if "tick_volume" in df else "volume" if "volume" in df else None
    if vc:
        recent=df.iloc[max(0,len(df)-min(100,cfg.pool_lookback)):]
        mask=recent.vol_sma_20.notna() & (recent[vc]>=cfg.volume_spike_multiple*recent.vol_sma_20) & ((recent.close-recent.open).abs()<=cfg.narrow_body_atr_multiple*recent.atr_14.replace(0,np.nan))
        for idx,row in recent.loc[mask].iterrows():
            atr=float(row.atr_14); w=max(0,mult*atr); level=float((row.high+row.low)/2)
            pools.append(LiquidityPool("ABSORPTION_NODE","BOTH",level,level-w,level+w,idx,"high_volume_narrow_body",float(row[vc]),{"bar_delta":float(row.bar_delta),"cvd":float(row.cvd)}))
    return pools

def btc_cvd_regime_filter(frame: pd.DataFrame, *, config: OrderFlowConfig | None = None) -> pd.Series:
    """Causal BTC leadership regime from rolling price and CVD direction.

    The regime is directional only when BTC price change and taker-delta CVD
    agree over the same lookback; ambiguous regimes intentionally fail closed.
    """
    cfg = config or OrderFlowConfig()
    features = calculate_orderflow_features(frame, config=cfg)
    if features.empty:
        return pd.Series(dtype="object", index=features.index, name="btc_regime")
    lookback = max(5, int(cfg.delta_lookback))
    price_change = features["close"].pct_change(lookback)
    cvd_change = features["bar_delta"].rolling(lookback, min_periods=lookback).sum()
    bull = (price_change > 0) & (cvd_change > 0)
    bear = (price_change < 0) & (cvd_change < 0)
    regime = pd.Series("NEUTRAL", index=features.index, dtype="object", name="btc_regime")
    regime.loc[bull] = "IMBALANCE_BULL"
    regime.loc[bear] = "IMBALANCE_BEAR"
    return regime


def _metrics(df):
    if df.empty: return {"bar_delta":0.0,"cvd_value":0.0,"delta_zscore":0.0}
    x=df.iloc[-1]
    return {"bar_delta":float(x.bar_delta),"cvd_value":float(x.cvd),"delta_zscore":float(x.delta_zscore),"delta_source":str(x.delta_source),"delta_quality":str(x.delta_quality) if "delta_quality" in x else ("REAL" if str(x.delta_source) in ("real_taker","tick_rule") else "ESTIMATED")}

def evaluate_orderflow_trigger(frame: pd.DataFrame, side: str, *, poi: LiquidityPool|Mapping[str,Any]|None=None, ticks: pd.DataFrame|None=None, config: OrderFlowConfig|None=None, session_allowed: bool=True, spread: float|None=None, stop_distance: float|None=None, tp1_distance: float|None=None, tp2_distance: float|None=None, min_net_rr_tp1: float=1.0, min_net_rr_tp2: float=1.5, max_spread_to_sl: float=0.25, strict_real_delta: bool|None=None) -> TriggerDecision:
    cfg=config or OrderFlowConfig(); side=str(side).upper()
    if side not in ("LONG","SHORT"): return TriggerDecision("NO_TRADE",reason="invalid_side")
    df=calculate_orderflow_features(frame,ticks=ticks,config=cfg)
    if len(df)<max(3,cfg.delta_lookback//2): return TriggerDecision("NO_TRADE",reason="insufficient_bars")
    strict = cfg.strict_real_delta if strict_real_delta is None else bool(strict_real_delta)
    if strict and str(df.delta_source.iloc[-1]) not in ("real_taker", "tick_rule"):
        return TriggerDecision("NO_TRADE", reason="strict_real_delta_required", metrics={**_metrics(df), "delta_quality": "ESTIMATED"})
    if not session_allowed: return TriggerDecision("NO_TRADE",reason="session_veto")
    if isinstance(df.index, pd.DatetimeIndex):
        hour = int(df.index[-1].hour)
        start_hour, end_hour = cfg.active_hours_utc
        active = (start_hour <= hour < end_hour) if start_hour < end_hour else (hour >= start_hour or hour < end_hour)
        if not active: return TriggerDecision("NO_TRADE", reason="inactive_liquidity_hours", metrics=_metrics(df))
    if poi is not None:
        if isinstance(poi,LiquidityPool): lo,hi=poi.lower,poi.upper
        else:
            level=float(poi.get("level",poi.get("price",0))); lo=float(poi.get("lower",poi.get("low",level))); hi=float(poi.get("upper",poi.get("high",level)))
        tail=df.iloc[-cfg.sweep_lookback:]
        if not (((tail.low<=hi)&(tail.high>=lo)).any() or ((tail.low<lo)|(tail.high>hi)).any()):
            return TriggerDecision("NO_TRADE",reason="poi_not_touched")
    # Use a prior local extremum and the penultimate bar as the candidate absorption bar;
    # the latest bar is the delta-flip confirmation. Only closed bars may be passed.
    prior_end=len(df)-2; prior_start=max(0,prior_end-cfg.divergence_lookback)
    prior=df.iloc[prior_start:prior_end]; candidate=df.iloc[-2]; trigger=df.iloc[-1]
    if prior.empty: return TriggerDecision("NO_TRADE",reason="insufficient_divergence_window",metrics=_metrics(df))
    if side=="LONG":
        pos=prior_start+int(np.argmin(prior.low.to_numpy())); extreme=float(df.low.iloc[pos]); cvd1=float(df.cvd.iloc[pos])
        cvd2=float(candidate.cvd); exhaustion=float(candidate.low)<=extreme and cvd2>cvd1
        poi_lower=(poi.lower if isinstance(poi,LiquidityPool) else (poi.get("lower",poi.get("low")) if poi else None))
        held=float(candidate.close)>extreme or (poi_lower is not None and float(candidate.close)>float(poi_lower))
        d=df.bar_delta.iloc[max(0,len(df)-4):-1]; avg=df.delta_abs_mean_20.iloc[max(0,len(df)-4):-1]
        aggressive_sell = bool((d <= -cfg.absorption_delta_multiple * avg).any())
        absorption = aggressive_sell and held
        flip = (float(trigger.bar_delta) >= cfg.flip_delta_multiple * float(trigger.delta_abs_mean_20)
                and float(trigger.close) > float(trigger.open))
    else:
        pos=prior_start+int(np.argmax(prior.high.to_numpy())); extreme=float(df.high.iloc[pos]); cvd1=float(df.cvd.iloc[pos])
        cvd2=float(candidate.cvd); exhaustion=float(candidate.high)>=extreme and cvd2<cvd1
        poi_upper=(poi.upper if isinstance(poi,LiquidityPool) else (poi.get("upper",poi.get("high")) if poi else None))
        held=float(candidate.close)<extreme or (poi_upper is not None and float(candidate.close)<float(poi_upper))
        d=df.bar_delta.iloc[max(0,len(df)-4):-1]; avg=df.delta_abs_mean_20.iloc[max(0,len(df)-4):-1]
        aggressive_buy = bool((d >= cfg.absorption_delta_multiple * avg).any())
        absorption = aggressive_buy and held
        flip = (float(trigger.bar_delta) <= -cfg.flip_delta_multiple * float(trigger.delta_abs_mean_20)
                and float(trigger.close) < float(trigger.open))
    kind=("ABSORPTION" if absorption else "EXHAUSTION" if exhaustion and cfg.allow_exhaustion else None)
    metrics={**_metrics(df),"cvd_divergence_type":kind,"prior_extreme_index":pos,"prior_cvd":cvd1,"candidate_cvd":cvd2}
    if kind is None: return TriggerDecision("NO_TRADE",reason="no_cvd_divergence_or_absorption",metrics=metrics)
    if not flip: return TriggerDecision("NO_TRADE",reason="delta_flip_missing",metrics=metrics)
    if spread is not None and stop_distance is not None and (stop_distance<=0 or spread<0 or spread/stop_distance>max_spread_to_sl):
        return TriggerDecision("NO_TRADE",reason="spread_to_sl_veto",metrics=metrics)
    if stop_distance is not None:
        if stop_distance<=0: return TriggerDecision("NO_TRADE",reason="invalid_stop_distance",metrics=metrics)
        if tp1_distance is not None and tp1_distance/stop_distance<min_net_rr_tp1: return TriggerDecision("NO_TRADE",reason="tp1_rr_veto",metrics=metrics)
        if tp2_distance is not None and tp2_distance/stop_distance<min_net_rr_tp2: return TriggerDecision("NO_TRADE",reason="tp2_rr_veto",metrics=metrics)
    return TriggerDecision("EXECUTE",side=side,reason="orderflow_confirmed",metrics=metrics)
