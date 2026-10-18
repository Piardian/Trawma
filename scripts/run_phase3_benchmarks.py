"""Run Phase 3 benchmarks:
1. 100k closed M1 bars on M15 & M5 for BTCUSDT and SOLUSDT (amt, orderflow, hybrid) using Binance data cache.
2. 50k closed bars on M15 & M5 for XAUUSD (orderflow, hybrid, amt) using open MT5 terminal.
3. Compare against Phase 2 results.
"""
from __future__ import annotations
import json
import math
import time
from pathlib import Path
import numpy as np
import pandas as pd

from pragma.data.binance_data import BinanceDataClient, filter_closed_bars as binance_filter_closed
from pragma.data.mt5_data import MT5DataClient, filter_closed_bars as mt5_filter_closed
from pragma.backtest_engine import simulate_walk_forward
from scripts.run_real_comparison import aggregate


def run_crypto_benchmarks(count: int = 100000, cache_dir: str = "data_cache"):
    print(f"\n=======================================================")
    print(f"--- 1. RUNNING BINANCE CRYPTO 100K BENCHMARKS (PHASE 3) ---")
    print(f"=======================================================")
    reports = []
    with BinanceDataClient(cache_dir=cache_dir, futures_only=True) as client:
        print(f"Loading {count} closed M1 bars for BTCUSDT from cache...")
        btc_m1 = client.rates("BTCUSDT", "M1", count=count)
        btc_m1 = binance_filter_closed(btc_m1, "M1")
        btc_by_tf = {
            "M15": aggregate(btc_m1, 15),
            "M5": aggregate(btc_m1, 5),
        }
        
        for symbol in ["BTCUSDT", "SOLUSDT"]:
            if symbol == "BTCUSDT":
                m1 = btc_m1
            else:
                print(f"Loading {count} closed M1 bars for {symbol} from cache...")
                m1 = client.rates(symbol, "M1", count=count)
                m1 = binance_filter_closed(m1, "M1")
            
            for tf, minutes in [("M15", 15), ("M5", 5)]:
                frame = aggregate(m1, minutes) if symbol != "BTCUSDT" else btc_by_tf[tf]
                frame = frame.copy()
                if "tick_volume" not in frame and "volume" in frame:
                    frame["tick_volume"] = frame["volume"]
                
                btc_frame = None if symbol == "BTCUSDT" else btc_by_tf[tf]
                
                for engine in ["amt", "orderflow", "hybrid"]:
                    strict = engine in ("orderflow", "hybrid")
                    t0 = time.time()
                    r = simulate_walk_forward(
                        frame,
                        symbol=symbol,
                        engine=engine,
                        strict_real_delta=strict,
                        btc_frame=btc_frame,
                    )
                    dt = time.time() - t0
                    r["timeframe"] = tf
                    r["bars_downloaded_m1"] = len(m1)
                    r["data_source"] = "binance_futures"
                    reports.append(r)
                    
                    pf_str = "inf" if math.isinf(r["profit_factor"]) else f"{r['profit_factor']:.2f}"
                    by_div = r.get("by_divergence_type", {})
                    abs_tr = by_div.get("ABSORPTION", {}).get("trades", 0)
                    exh_tr = by_div.get("EXHAUSTION", {}).get("trades", 0)
                    print(
                        f"[{symbol} {tf:3s} {engine:9s}] Trades: {r['total_trades']:3d} | "
                        f"WR: {r['win_rate']*100:5.1f}% | PF: {pf_str:>5s} | Net R: {r['net_r']:+6.2f}R | "
                        f"Max DD: {r['max_drawdown_r']:5.2f}R | Abs: {abs_tr:2d} | Exh: {exh_tr:2d} ({dt:.1f}s)",
                        flush=True,
                    )
    return reports


def run_mt5_benchmarks(count: int = 50000):
    print(f"\n=======================================================")
    print(f"--- 2. RUNNING MT5 XAUUSD 50K BENCHMARKS (PHASE 3) ---")
    print(f"=======================================================")
    reports = []
    symbol = "XAUUSD"
    with MT5DataClient() as client:
        for tf in ["M15", "M5"]:
            print(f"Fetching {count} closed bars for {symbol} {tf} from MT5...")
            raw_bars = client.rates(symbol, tf, count=count)
            bars = mt5_filter_closed(raw_bars, tf)
            bars = bars.copy()
            print(f"Received {len(bars)} closed {tf} bars for {symbol}.")
            
            for engine in ["amt", "orderflow", "hybrid"]:
                t0 = time.time()
                # MT5 rates provide tick_volume with synthetic BVC delta; strict_real_delta is False for BVC
                r = simulate_walk_forward(
                    bars,
                    symbol=symbol,
                    engine=engine,
                    strict_real_delta=False,
                )
                dt = time.time() - t0
                r["timeframe"] = tf
                r["data_source"] = "mt5"
                reports.append(r)
                
                pf_str = "inf" if math.isinf(r["profit_factor"]) else f"{r['profit_factor']:.2f}"
                by_div = r.get("by_divergence_type", {})
                abs_tr = by_div.get("ABSORPTION", {}).get("trades", 0)
                exh_tr = by_div.get("EXHAUSTION", {}).get("trades", 0)
                print(
                    f"[{symbol} {tf:3s} {engine:9s}] Trades: {r['total_trades']:3d} | "
                    f"WR: {r['win_rate']*100:5.1f}% | PF: {pf_str:>5s} | Net R: {r['net_r']:+6.2f}R | "
                    f"Max DD: {r['max_drawdown_r']:5.2f}R | Abs: {abs_tr:2d} | Exh: {exh_tr:2d} ({dt:.1f}s)",
                    flush=True,
                )
    return reports


def main():
    crypto_reports = run_crypto_benchmarks(count=100000)
    mt5_reports = run_mt5_benchmarks(count=50000)
    all_reports = crypto_reports + mt5_reports
    
    out = Path("artifacts/phase3_walk_forward_comparison.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(all_reports, indent=2, allow_nan=True, default=str) + "\n", encoding="utf-8")
    print(f"\nPhase 3 results written to {out}")

    # Load Phase 2 results for comparison
    phase2_file = Path("artifacts/real_walk_forward_100k_comparison.json")
    phase2_map = {}
    if phase2_file.exists():
        with open(phase2_file, encoding="utf-8") as f:
            p2_data = json.load(f)
            for d in p2_data:
                key = (d["symbol"], d.get("timeframe", ""), d["engine"])
                phase2_map[key] = d

    print("\n=======================================================")
    print("--- 3. COMPARISON TABLE: PHASE 2 vs PHASE 3 ---")
    print("=======================================================\n")
    print("| Symbol | TF | Engine | Source | P2 Trades | P3 Trades | P2 Net R | P3 Net R | P2 WR | P3 WR | P2 PF | P3 PF | P2 MaxDD | P3 MaxDD | P3 Abs/Exh |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    
    for r in all_reports:
        sym = r["symbol"]
        tf = r.get("timeframe", "")
        eng = r["engine"]
        src = r.get("data_source", "binance")
        p2 = phase2_map.get((sym, tf, eng))
        
        p2_trades = f"{p2['total_trades']}" if p2 else "-"
        p3_trades = f"{r['total_trades']}"
        
        p2_netr = f"{p2['net_r']:+.2f}R" if p2 else "-"
        p3_netr = f"{r['net_r']:+.2f}R"
        
        p2_wr = f"{p2['win_rate']*100:.1f}%" if p2 else "-"
        p3_wr = f"{r['win_rate']*100:.1f}%"
        
        p2_pf = ("inf" if math.isinf(p2['profit_factor']) else f"{p2['profit_factor']:.2f}") if p2 and p2['profit_factor'] is not None else "-"
        p3_pf = ("inf" if math.isinf(r['profit_factor']) else f"{r['profit_factor']:.2f}") if r['profit_factor'] is not None else "-"
        
        p2_dd = f"{p2['max_drawdown_r']:.2f}R" if p2 else "-"
        p3_dd = f"{r['max_drawdown_r']:.2f}R"
        
        by_div = r.get("by_divergence_type", {})
        abs_tr = by_div.get("ABSORPTION", {}).get("trades", 0)
        exh_tr = by_div.get("EXHAUSTION", {}).get("trades", 0)
        div_str = f"{abs_tr}/{exh_tr}"
        
        print(f"| {sym} | {tf} | {eng} | {src} | {p2_trades} | {p3_trades} | {p2_netr} | {p3_netr} | {p2_wr} | {p3_wr} | {p2_pf} | {p3_pf} | {p2_dd} | {p3_dd} | {div_str} |")


if __name__ == "__main__":
    main()
