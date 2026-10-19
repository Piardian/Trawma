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
    evaluate_orderflow_trigger, evaluate_cvd_momentum_trigger, btc_cvd_regime_filter,
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
    max_hold_bars: int = 16

def _metrics(trades: list[dict], funnel: dict, symbol: str, engine: str) -> dict[str, Any]:
    rs=np.array([t["net_r"] for t in trades],dtype=float)
    wins=rs[rs>0]; losses=rs[rs<0]
    pf=float(wins.sum()/abs(losses.sum())) if len(losses) and abs(losses.sum()) else (None if len(wins) else 0.0)
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
        for k in ("ABSORPTION","EXHAUSTION","CVD_MOMENTUM_PULLBACK")
      },
      "trades":trades,
    }

def simulate_walk_forward(frame: pd.DataFrame, *, symbol: str="UNKNOWN",
                          engine: str="orderflow", strict_real_delta: bool=False,
                          fee_bps_per_side: float=4.0, funding_rate: float=0.0001,
                          atr_multiple: float=1.0, rr_tp1: float=1.0, rr_tp2: float=2.0,
                          config: OrderFlowConfig|None=None, ticks: pd.DataFrame|None=None,
                          btc_frame: pd.DataFrame|None=None,
                          macro_bias: int|None=None) -> dict[str,Any]:
    valid_engines = {"amt", "orderflow", "orderflow_absorption", "hybrid", "hybrid_absorption", "cvd_momentum"}
    if engine not in valid_engines:
        raise ValueError(f"engine must be one of {sorted(valid_engines)}")
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
    features=calculate_orderflow_features(df,ticks=ticks,config=cfg)
    funnel = {
        "bars_seen": 0, "poi_touch": 0, "divergence": 0, "delta_flip": 0, "cost_veto": 0,
        "entries": 0, "stop_exits": 0, "tp1": 0, "tp2": 0, "time_stop": 0,
        "weekend_close": 0, "weekend_entry_veto": 0, "macro_bias_veto": 0, "end_of_data": 0,
        "strict_delta_veto": 0, "exhaustion_veto": 0, "inactive_hours_veto": 0,
        "low_volume_day_veto": 0, "btc_regime_veto": 0,
        "bars_evaluated": 0, "veto_session_hours": 0, "veto_low_volume_day": 0,
        "veto_btc_leader": 0, "poi_or_pullback_touches": 0, "passed_absorption_or_cvd": 0,
        "passed_delta_flip_or_ignition": 0, "rejected_high_cost_to_sl": 0,
        "rejected_low_rr": 0, "trades_opened": 0,
    }
    trades=[]; pos=None
    fee=max(0.0,fee_bps_per_side)/10000.0
    # Causal intraday activity guard: compare cumulative volume pace with the
    # previous five completed UTC days; never use the eventual current-day total.
    day_allowed = pd.Series(True, index=df.index)
    if cfg.filter_low_volume_days and isinstance(df.index, pd.DatetimeIndex):
        day_keys = pd.Series(df.index.normalize(), index=df.index)
        if "volume" in df: volume_col = "volume"
        elif "tick_volume" in df: volume_col = "tick_volume"
        else: volume_col = None
        if volume_col:
            daily = df[volume_col].groupby(day_keys).sum()
            baseline = daily.shift(1).rolling(5, min_periods=5).mean()
            day_allowed[:] = False
            for day in daily.index:
                rows = df.index.normalize() == day
                if day.weekday() >= 5 or pd.isna(baseline.loc[day]) or baseline.loc[day] <= 0:
                    continue
                cumulative = df.loc[rows, volume_col].fillna(0).clip(lower=0).cumsum()
                elapsed_fraction = np.clip((df.index[rows] - day).total_seconds() / 86400.0, 0.0, 1.0)
                pace_threshold = 0.70 * float(baseline.loc[day]) * elapsed_fraction
                day_allowed.loc[rows] = cumulative.to_numpy() >= pace_threshold
    btc_regimes = None
    if btc_frame is not None:
        raw_regimes = btc_cvd_regime_filter(btc_frame, config=cfg)
        if not raw_regimes.empty and isinstance(df.index, pd.DatetimeIndex):
            # Causal asof alignment: each bar timestamp gets the latest regime up to that point
            btc_regimes = pd.Series(
                pd.merge_asof(
                    pd.DataFrame(index=df.index),
                    raw_regimes.to_frame("regime"),
                    left_index=True,
                    right_index=True,
                    direction="backward",
                )["regime"].fillna("NEUTRAL").values,
                index=df.index,
            )
        else:
            btc_regimes = raw_regimes
    # Causal precomputation for hybrid engine: prior-day completed Volume Profiles (pdVAH, pdVAL, pdPOC)
    prior_session_profiles = {}
    rolling_100_std = None
    if engine in ("hybrid", "hybrid_absorption"):
        from .amt_engine import calculate_volume_profile
        rolling_100_std = df["close"].rolling(100, min_periods=20).std(ddof=0).fillna(0.0).to_numpy()
        if isinstance(df.index, pd.DatetimeIndex):
            unique_days = list(pd.unique(pd.Series(df.index.date, index=df.index)))
            for day_idx in range(1, len(unique_days)):
                prev_d = unique_days[day_idx - 1]
                cur_d = unique_days[day_idx]
                prev_bars = df.loc[df.index.date == prev_d]
                if len(prev_bars) >= 10:
                    try:
                        vc = "tick_volume" if "tick_volume" in prev_bars else "volume"
                        vp = calculate_volume_profile(prev_bars[["high", "low", "close", vc]], bins=50)
                        prior_session_profiles[cur_d] = (vp.val, vp.vah, vp.poc)
                    except Exception:
                        pass
    normalized_symbol = symbol.upper().replace("/", "").replace("-", "").replace("_", "")
    is_alt = normalized_symbol.endswith("USDT") and normalized_symbol not in {"BTCUSDT"}
    is_crypto = normalized_symbol.endswith("USDT")
    for i in range(50,len(df)-1):
        bar=df.iloc[i]; nextbar=df.iloc[i+1]; ts=df.index[i]
        funnel["bars_seen"]+=1
        funnel["bars_evaluated"]+=1
        # Existing position management first; stop wins if SL and TP share a bar.
        if pos is not None:
            # Weekend gap protection: Friday close before market shut
            if not is_crypto and isinstance(ts, pd.Timestamp) and ts.weekday() == 4:
                h_exit, m_exit = cfg.friday_exit_cutoff_utc
                if ts.hour > h_exit or (ts.hour == h_exit and ts.minute >= m_exit):
                    exit_price = float(bar.close)
                    gross = ((exit_price - pos.entry) if pos.side == "LONG" else (pos.entry - exit_price)) / pos.risk
                    pos.realized_r += pos.qty_remaining * gross - fee * (pos.entry + exit_price) * pos.qty_remaining / pos.risk
                    trades.append({"entry_time": str(df.index[pos.entry_index]), "exit_time": str(ts), "side": pos.side, "net_r": pos.realized_r, "exit_reason": "WEEKEND_CLOSE", "tp1_taken": pos.tp1_taken, "divergence_type": pos.divergence})
                    funnel["weekend_close"] += 1; pos = None; continue

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
                funnel["tp2"]+=1; pos=None; continue

            # Time-Stop exit: if TP1 not reached within max_hold_bars, exit at current close
            if not pos.tp1_taken and (i - pos.entry_index) >= pos.max_hold_bars:
                exit_price = float(bar.close)
                gross = ((exit_price - pos.entry) if pos.side == "LONG" else (pos.entry - exit_price)) / pos.risk
                pos.realized_r += pos.qty_remaining * gross - fee * (pos.entry + exit_price) * pos.qty_remaining / pos.risk
                trades.append({"entry_time": str(df.index[pos.entry_index]), "exit_time": str(ts), "side": pos.side, "net_r": pos.realized_r, "exit_reason": "TIME_STOP", "tp1_taken": pos.tp1_taken, "divergence_type": pos.divergence})
                funnel["time_stop"] += 1; pos = None; continue

            if pos is not None: continue
        # Session and liquidity gates are evaluated before any new signal.
        if isinstance(df.index, pd.DatetimeIndex):
            # Friday entry cutoff: prevent new positions before weekend close for non-crypto
            if not is_crypto and ts.weekday() == 4:
                h_entry, m_entry = cfg.friday_entry_cutoff_utc
                if ts.hour > h_entry or (ts.hour == h_entry and ts.minute >= m_entry):
                    funnel["weekend_entry_veto"] += 1
                    continue
            hour = int(ts.hour)
            h0, h1 = cfg.active_hours_utc
            if not (h0 <= hour < h1 if h0 < h1 else hour >= h0 or hour < h1):
                funnel["inactive_hours_veto"] += 1
                funnel["veto_session_hours"] += 1
                continue
            if not bool(day_allowed.iloc[i]):
                funnel["low_volume_day_veto"] += 1
                funnel["veto_low_volume_day"] += 1
                continue
        # Bounded causal window prevents quadratic recomputation on 100k-bar runs.
        prefix=features.iloc[max(0, i-max(cfg.pool_lookback, cfg.delta_lookback*2, 100)):i+1]
        atr=float(features.atr_14.iloc[i])
        if not np.isfinite(atr) or atr<=0: continue
        side=None; decision=None; div="UNKNOWN"
        pos_risk=None; pos_max_hold=cfg.max_hold_bars; pos_rr_tp1=rr_tp1; pos_rr_tp2=rr_tp2

        if engine in ("orderflow", "orderflow_absorption"):
            poi=None
            pools=detect_orderflow_pools(prefix,config=cfg)
            touched=[p for p in pools if float(bar.low)<=p.upper and float(bar.high)>=p.lower]
            if touched:
                funnel["poi_touch"]+=1
                funnel["poi_or_pullback_touches"]+=1
                pool=next((p for p in reversed(touched) if p.side in ("LONG","SHORT")), None)
                if pool is not None:
                    side=pool.side
                    poi=pool
            if side:
                decision=evaluate_orderflow_trigger(prefix,side,poi=poi,ticks=(ticks.loc[(ticks.index >= prefix.index[0]) & (ticks.index <= prefix.index[-1])] if ticks is not None and isinstance(ticks.index,pd.DatetimeIndex) else None),config=cfg,strict_real_delta=strict_real_delta)
                if decision.action=="EXECUTE":
                    div=str(decision.metrics.get("cvd_divergence_type") or "UNKNOWN")
                    funnel["divergence"]+=1
                    funnel["delta_flip"]+=1
                    funnel["passed_absorption_or_cvd"]+=1
                    funnel["passed_delta_flip_or_ignition"]+=1
                else:
                    if decision.reason=="strict_real_delta_required": funnel["strict_delta_veto"]+=1
                    if decision.reason=="exhaustion_disabled": funnel["exhaustion_veto"]+=1
                    side=None

        elif engine in ("hybrid", "hybrid_absorption"):
            poi=None
            try:
                cur_date = ts.date() if isinstance(ts, pd.Timestamp) else None
                pd_prof = prior_session_profiles.get(cur_date) if cur_date else None
                vwap_val = float(prefix["vwap"].iloc[-1])
                vwap_std = float(rolling_100_std[i]) if rolling_100_std is not None else 0.0
                candidates = [
                    ("LONG", vwap_val), ("SHORT", vwap_val),
                    ("LONG", vwap_val - 1.5 * vwap_std), ("SHORT", vwap_val + 1.5 * vwap_std)
                ]
                if pd_prof is not None:
                    val_lvl, vah_lvl, poc_lvl = pd_prof
                    candidates.extend([
                        ("LONG", val_lvl), ("SHORT", vah_lvl),
                        ("LONG", poc_lvl), ("SHORT", poc_lvl)
                    ])
                hit = [(s, l) for s, l in candidates if abs(float(bar.close) - l) <= 0.35 * atr]
                if hit:
                    funnel["poi_touch"] += 1
                    funnel["poi_or_pullback_touches"] += 1
                    side, level = hit[-1]
                    poi = {"level": level, "lower": level - 0.25 * atr, "upper": level + 0.25 * atr}
            except (ValueError, KeyError, TypeError):
                side = None
            if side:
                decision=evaluate_orderflow_trigger(prefix,side,poi=poi,ticks=(ticks.loc[(ticks.index >= prefix.index[0]) & (ticks.index <= prefix.index[-1])] if ticks is not None and isinstance(ticks.index,pd.DatetimeIndex) else None),config=cfg,strict_real_delta=strict_real_delta)
                if decision.action=="EXECUTE":
                    div=str(decision.metrics.get("cvd_divergence_type") or "UNKNOWN")
                    funnel["divergence"]+=1
                    funnel["delta_flip"]+=1
                    funnel["passed_absorption_or_cvd"]+=1
                    funnel["passed_delta_flip_or_ignition"]+=1
                else:
                    if decision.reason=="strict_real_delta_required": funnel["strict_delta_veto"]+=1
                    if decision.reason=="exhaustion_disabled": funnel["exhaustion_veto"]+=1
                    side=None

        elif engine == "cvd_momentum":
            cur_prev = prefix.iloc[-2] if len(prefix) >= 2 else bar
            pb_support = max(float(cur_prev["ema_20"]), float(cur_prev["vwap"]))
            pb_resist = min(float(cur_prev["ema_20"]), float(cur_prev["vwap"]))
            pb_window = prefix.iloc[max(0, len(prefix) - 6): -1]
            if len(pb_window) >= 3:
                if (pb_window.low <= pb_support + 0.35 * atr).any() or (pb_window.high >= pb_resist - 0.35 * atr).any():
                    funnel["poi_touch"] += 1
                    funnel["poi_or_pullback_touches"] += 1

            decision = evaluate_cvd_momentum_trigger(
                prefix,
                ticks=(ticks.loc[(ticks.index >= prefix.index[0]) & (ticks.index <= prefix.index[-1])] if ticks is not None and isinstance(ticks.index, pd.DatetimeIndex) else None),
                config=cfg, strict_real_delta=strict_real_delta, fee_bps_per_side=fee_bps_per_side,
            )
            if decision.action == "EXECUTE":
                side = decision.side
                div = str(decision.metrics.get("cvd_divergence_type") or "CVD_MOMENTUM_PULLBACK")
                pos_risk = float(decision.metrics.get("risk", atr * 1.2))
                pos_max_hold = int(decision.metrics.get("max_hold_bars", 20))
                pos_rr_tp1 = float(decision.metrics.get("rr_tp1", 1.3))
                pos_rr_tp2 = float(decision.metrics.get("rr_tp2", 2.5))
                funnel["divergence"] += 1
                funnel["delta_flip"] += 1
                funnel["passed_absorption_or_cvd"] += 1
                funnel["passed_delta_flip_or_ignition"] += 1
            else:
                if decision.reason == "strict_real_delta_required": funnel["strict_delta_veto"] += 1
                side = None

        else:
            # AMT-only control: causal z-score excursion from rolling typical-price mean.
            typical=(prefix.high+prefix.low+prefix.close)/3.0
            mean=typical.rolling(50,min_periods=20).mean().iloc[-1]
            std=typical.rolling(50,min_periods=20).std(ddof=0).iloc[-1]
            z=(float(bar.close)-float(mean))/float(std) if pd.notna(std) and std>0 else 0.0
            if z<=-2.0: side="LONG"
            elif z>=2.0: side="SHORT"
        if side not in ("LONG","SHORT"): continue
        if macro_bias is not None:
            if (macro_bias == 1 and side != "LONG") or (macro_bias == -1 and side != "SHORT") or (macro_bias == 0):
                funnel["macro_bias_veto"] += 1
                continue
        if is_alt and engine in ("orderflow", "orderflow_absorption", "hybrid", "hybrid_absorption", "cvd_momentum"):
            if btc_regimes is None:
                funnel["btc_regime_veto"] += 1
                funnel["veto_btc_leader"] += 1
                continue
            regime = str(btc_regimes.iloc[i]) if btc_regimes is not None else "NEUTRAL"
            if not ((side == "LONG" and regime == "IMBALANCE_BULL") or
                    (side == "SHORT" and regime == "IMBALANCE_BEAR")):
                funnel["btc_regime_veto"] += 1
                funnel["veto_btc_leader"] += 1
                continue
        entry=float(nextbar.open)
        risk=pos_risk if pos_risk is not None else max(atr*atr_multiple,entry*fee*2.5)
        stop=entry-risk if side=="LONG" else entry+risk
        tp1=entry+pos_rr_tp1*risk*(1 if side=="LONG" else -1)
        tp2=entry+pos_rr_tp2*risk*(1 if side=="LONG" else -1)
        # Net-cost guard: spread/fee cost must not consume too much of the initial risk.
        if (2*fee*entry)/risk>0.25:
            funnel["cost_veto"]+=1
            funnel["rejected_high_cost_to_sl"]+=1
            continue
        pos=Position(side,entry,stop,tp1,tp2,risk,i+1,div,max_hold_bars=pos_max_hold)
        funnel["entries"]+=1
        funnel["trades_opened"]+=1
    if pos is not None:
        last=df.iloc[-1]; ts=df.index[-1]; exit_price=float(last.close)
        gross=((exit_price-pos.entry) if pos.side=="LONG" else (pos.entry-exit_price))/pos.risk
        pos.realized_r += pos.qty_remaining*gross - fee*(pos.entry+exit_price)*pos.qty_remaining/pos.risk
        trades.append({"entry_time":str(df.index[pos.entry_index]),"exit_time":str(ts),"side":pos.side,"net_r":pos.realized_r,"exit_reason":"END_OF_DATA","tp1_taken":pos.tp1_taken,"divergence_type":pos.divergence})
        funnel["end_of_data"]+=1
    report=_metrics(trades,funnel,symbol,engine)
    report["bars_processed"]=len(df)
    report["absorption_only"] = not cfg.allow_exhaustion
    report["active_hours_utc"] = list(cfg.active_hours_utc)
    report["low_volume_day_filter"] = cfg.filter_low_volume_days
    report["btc_cvd_regime_filter"] = bool(is_alt and engine in ("orderflow", "hybrid"))
    report["delta_source"]=str(features["delta_source"].iloc[-1]) if len(features) else "empty"
    return report
