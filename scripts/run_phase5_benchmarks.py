"""Run Phase 5 Final Benchmarks:
- Time-Stop (16 bars) and Friday Weekend Gap (20:30 entry veto, 21:30 close)
- Instruments:
  1. XAUUSD (MT5, M5, 50k bars)
  2. NAS100 (MT5, M5 & M15, 50k bars)
  3. SP500 (MT5, M5 & M15, 50k bars)
  4. BTCUSDT (Binance, M15, 100k M1 bars)
  5. SOLUSDT (Binance, M15, 100k M1 bars, BTC CVD gated)
  6. EURUSD with Begonya Macro Bias = -1 (SHORT ONLY) (MT5, M5 & M15, 50k bars)
"""
from __future__ import annotations
import json
import math
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import pandas as pd

from pragma.data.binance_data import BinanceDataClient, filter_closed_bars as binance_filter_closed
from pragma.data.mt5_data import MT5DataClient, filter_closed_bars as mt5_filter_closed
from pragma.backtest_engine import simulate_walk_forward
from scripts.run_real_comparison import aggregate


def run_single_simulation(args):
    symbol, tf, engine, bars, fee_bps, strict, btc_frame, macro_bias = args
    t0 = time.time()
    res = simulate_walk_forward(
        bars,
        symbol=symbol,
        engine=engine,
        strict_real_delta=strict,
        fee_bps_per_side=fee_bps,
        funding_rate=0.0001 if symbol.endswith("USDT") else 0.0,
        btc_frame=btc_frame,
        macro_bias=macro_bias,
    )
    dt = time.time() - t0
    res["timeframe"] = tf
    res["macro_bias_applied"] = macro_bias
    pf_str = "inf" if math.isinf(res["profit_factor"]) else (f"{res['profit_factor']:.2f}" if res['profit_factor'] is not None else "0.00")
    by_div = res.get("by_divergence_type", {})
    abs_tr = by_div.get("ABSORPTION", {}).get("trades", 0)
    fc = res.get("funnel_counts", {})
    t_stops = fc.get("time_stop", 0)
    wk_closes = fc.get("weekend_close", 0)
    
    bias_str = f"BIAS:{macro_bias:+d}" if macro_bias is not None else "UNANCHORED"
    print(
        f"[{symbol:7s} {tf:3s} {bias_str:11s}] Trades: {res['total_trades']:3d} | "
        f"WR: {res['win_rate']*100:5.1f}% | PF: {pf_str:>5s} | Net R: {res['net_r']:+6.2f}R | "
        f"MaxDD: {res['max_drawdown_r']:5.2f}R | TimeStops: {t_stops:2d} | WkCloses: {wk_closes:2d} ({dt:.1f}s)",
        flush=True,
    )
    return res


def main():
    print("================================================================")
    print("--- 1. LOADING DATA FOR PHASE 5 FINAL BENCHMARKS ---")
    print("================================================================")
    
    # 1. Fetch MT5 symbols: XAUUSD, NAS100, SP500, EURUSD
    mt5_data = {}
    mt5_targets = ["XAUUSD", "NAS100", "SP500", "EURUSD"]
    with MT5DataClient() as client:
        avail = client.symbols()
        for sym in mt5_targets:
            if sym not in avail:
                print(f"[Warning] {sym} not in MT5 symbols, skipping")
                continue
            mt5_data[sym] = {}
            for tf in ["M5", "M15"]:
                raw = client.rates(sym, tf, count=50000)
                closed = mt5_filter_closed(raw, tf)
                mt5_data[sym][tf] = closed.copy()
                print(f"  -> MT5 {sym} {tf}: {len(closed)} closed bars.")

    # 2. Fetch Binance crypto: BTCUSDT, SOLUSDT (100k M1 -> M15)
    crypto_data = {}
    with BinanceDataClient(cache_dir="data_cache", futures_only=True) as bnc:
        btc_m1 = bnc.rates("BTCUSDT", "M1", count=100000)
        btc_m1 = binance_filter_closed(btc_m1, "M1")
        btc_m15 = aggregate(btc_m1, 15)
        crypto_data["BTCUSDT"] = btc_m15.copy()
        print(f"  -> Binance BTCUSDT M15: {len(btc_m15)} closed bars.")

        sol_m1 = bnc.rates("SOLUSDT", "M1", count=100000)
        sol_m1 = binance_filter_closed(sol_m1, "M1")
        sol_m15 = aggregate(sol_m1, 15)
        crypto_data["SOLUSDT"] = sol_m15.copy()
        print(f"  -> Binance SOLUSDT M15: {len(sol_m15)} closed bars.")

    tasks = []

    # XAUUSD M5
    if "XAUUSD" in mt5_data:
        tasks.append(("XAUUSD", "M5", "orderflow", mt5_data["XAUUSD"]["M5"], 1.0, False, None, None))

    # NAS100 M5 & M15
    if "NAS100" in mt5_data:
        tasks.append(("NAS100", "M5", "orderflow", mt5_data["NAS100"]["M5"], 1.0, False, None, None))
        tasks.append(("NAS100", "M15", "orderflow", mt5_data["NAS100"]["M15"], 1.0, False, None, None))

    # SP500 M5 & M15
    if "SP500" in mt5_data:
        tasks.append(("SP500", "M5", "orderflow", mt5_data["SP500"]["M5"], 1.0, False, None, None))
        tasks.append(("SP500", "M15", "orderflow", mt5_data["SP500"]["M15"], 1.0, False, None, None))

    # BTCUSDT M15
    tasks.append(("BTCUSDT", "M15", "orderflow", crypto_data["BTCUSDT"], 4.0, True, None, None))

    # SOLUSDT M15 (gated by BTC M15)
    tasks.append(("SOLUSDT", "M15", "orderflow", crypto_data["SOLUSDT"], 4.0, True, crypto_data["BTCUSDT"], None))

    # EURUSD M5 & M15 with Begonya Macro Bias = -1 (SHORT ONLY)
    if "EURUSD" in mt5_data:
        tasks.append(("EURUSD", "M5", "orderflow", mt5_data["EURUSD"]["M5"], 0.5, False, None, -1))
        tasks.append(("EURUSD", "M15", "orderflow", mt5_data["EURUSD"]["M15"], 0.5, False, None, -1))

    print(f"\nRunning {len(tasks)} Phase 5 simulation tasks across parallel workers...\n")

    results = []
    with ProcessPoolExecutor(max_workers=6) as executor:
        futures = {executor.submit(run_single_simulation, task): task for task in tasks}
        for future in as_completed(futures):
            results.append(future.result())

    results.sort(key=lambda r: (r["symbol"], r["timeframe"], str(r.get("macro_bias_applied", ""))))

    out = Path("artifacts/phase5_final_comparison.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, allow_nan=True, default=str) + "\n", encoding="utf-8")
    print(f"\nResults saved to {out}\n")

    print("================================================================")
    print("--- 2. PHASE 5 FINAL COMPARATIVE BENCHMARK RESULTS ---")
    print("================================================================\n")
    print("| Symbol | TF | Macro Anchor | Bars | Trades | Win Rate | Profit Factor | Net R | Exp R | Max DD R | Time-Stops | Wk-Closes |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")

    for r in results:
        sym = r["symbol"]
        tf = r["timeframe"]
        bias = f"BIAS: {r['macro_bias_applied']:+d}" if r.get("macro_bias_applied") is not None else "UNANCHORED"
        bars = r.get("bars_processed", 0)
        trades = r["total_trades"]
        wr = f"{r['win_rate']*100:.1f}%"
        pf = ("inf" if math.isinf(r["profit_factor"]) else f"{r['profit_factor']:.2f}") if r["profit_factor"] is not None else "-"
        net_r = f"{r['net_r']:+.2f}R"
        exp_r = f"{r['expectancy_r']:+.2f}R"
        max_dd = f"{r['max_drawdown_r']:.2f}R"
        fc = r.get("funnel_counts", {})
        t_stops = fc.get("time_stop", 0)
        wk_closes = fc.get("weekend_close", 0)
        print(f"| {sym} | {tf} | {bias} | {bars} | {trades} | {wr} | {pf} | {net_r} | {exp_r} | {max_dd} | {t_stops} | {wk_closes} |")


if __name__ == "__main__":
    main()
