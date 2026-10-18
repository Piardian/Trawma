"""Run Phase 4 FX benchmarks on 6 Forex pairs (50,000 closed bars on M5 & M15).
Pairs: EURUSD, CADCHF, USDJPY, CADJPY, EURGBP, GBPJPY
Engines: orderflow, hybrid
Spread cost: 10 points / 1 pip (fee_bps_per_side=0.5, funding_rate=0.0)
"""
from __future__ import annotations
import json
import math
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import pandas as pd

from pragma.data.mt5_data import MT5DataClient, filter_closed_bars
from pragma.backtest_engine import simulate_walk_forward

FX_SYMBOLS = ["EURUSD", "CADCHF", "USDJPY", "CADJPY", "EURGBP", "GBPJPY"]
TIMEFRAMES = ["M5", "M15"]
ENGINES = ["orderflow", "hybrid"]
COUNT = 50000
FEE_BPS = 0.5  # 10 points / 1 pip round-trip spread
FUNDING = 0.0


def run_single_simulation(args):
    symbol, tf, engine, bars = args
    t0 = time.time()
    res = simulate_walk_forward(
        bars,
        symbol=symbol,
        engine=engine,
        strict_real_delta=False,
        fee_bps_per_side=FEE_BPS,
        funding_rate=FUNDING,
    )
    dt = time.time() - t0
    res["timeframe"] = tf
    res["data_source"] = "mt5"
    pf_str = "inf" if math.isinf(res["profit_factor"]) else (f"{res['profit_factor']:.2f}" if res['profit_factor'] is not None else "0.00")
    by_div = res.get("by_divergence_type", {})
    abs_tr = by_div.get("ABSORPTION", {}).get("trades", 0)
    exh_tr = by_div.get("EXHAUSTION", {}).get("trades", 0)
    print(
        f"[{symbol:6s} {tf:3s} {engine:9s}] Trades: {res['total_trades']:3d} | "
        f"WR: {res['win_rate']*100:5.1f}% | PF: {pf_str:>5s} | Net R: {res['net_r']:+6.2f}R | "
        f"Exp: {res['expectancy_r']:+5.2f}R | MaxDD: {res['max_drawdown_r']:5.2f}R | "
        f"Abs: {abs_tr:2d} | Exh: {exh_tr:2d} ({dt:.1f}s)",
        flush=True,
    )
    return res


def main():
    print("================================================================")
    print("--- 1. FETCHING 50K CLOSED BARS FROM MT5 FOR 6 FX PAIRS ---")
    print("================================================================")
    data_dict = {}
    with MT5DataClient() as client:
        for sym in FX_SYMBOLS:
            data_dict[sym] = {}
            for tf in TIMEFRAMES:
                print(f"Fetching {COUNT} {tf} bars for {sym}...", flush=True)
                raw = client.rates(sym, tf, count=COUNT)
                closed = filter_closed_bars(raw, tf)
                data_dict[sym][tf] = closed.copy()
                print(f"  -> {sym} {tf}: {len(closed)} closed bars.")

    tasks = []
    for sym in FX_SYMBOLS:
        for tf in TIMEFRAMES:
            bars = data_dict[sym][tf]
            for eng in ENGINES:
                tasks.append((sym, tf, eng, bars))

    print(f"\nTotal tasks to run: {len(tasks)} (6 symbols x 2 timeframes x 2 engines)")
    print("Running parallel walk-forward simulations across 6 workers...\n")

    results = []
    with ProcessPoolExecutor(max_workers=6) as executor:
        futures = {executor.submit(run_single_simulation, task): task for task in tasks}
        for future in as_completed(futures):
            res = future.result()
            results.append(res)

    # Sort results cleanly by symbol, timeframe, engine
    results.sort(key=lambda r: (r["symbol"], r["timeframe"], r["engine"]))

    out = Path("artifacts/phase4_fx_orderflow_comparison.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, allow_nan=True, default=str) + "\n", encoding="utf-8")
    print(f"\nAll results saved to {out}\n")

    print("================================================================")
    print("--- 2. MT5 FOREX 50K ORDER FLOW (ABSORPTION-ONLY) RESULTS ---")
    print("================================================================\n")
    print("| Symbol | TF | Engine | Bars | Trades | Win Rate | Profit Factor | Net R | Expectancy R | Max DD R | t-stat | Absorption / Exhaustion |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")

    for r in results:
        sym = r["symbol"]
        tf = r["timeframe"]
        eng = r["engine"]
        bars = r.get("bars_processed", 0)
        trades = r["total_trades"]
        wr = f"{r['win_rate']*100:.1f}%"
        pf = ("inf" if math.isinf(r["profit_factor"]) else f"{r['profit_factor']:.2f}") if r["profit_factor"] is not None else "-"
        net_r = f"{r['net_r']:+.2f}R"
        exp_r = f"{r['expectancy_r']:+.2f}R"
        max_dd = f"{r['max_drawdown_r']:.2f}R"
        t_stat = f"{r['t_stat']:.2f}"
        by_div = r.get("by_divergence_type", {})
        abs_tr = by_div.get("ABSORPTION", {}).get("trades", 0)
        exh_tr = by_div.get("EXHAUSTION", {}).get("trades", 0)
        div_str = f"{abs_tr} / {exh_tr}"
        print(f"| {sym} | {tf} | {eng} | {bars} | {trades} | {wr} | {pf} | {net_r} | {exp_r} | {max_dd} | {t_stat} | {div_str} |")


if __name__ == "__main__":
    main()
