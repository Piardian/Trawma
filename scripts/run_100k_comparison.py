"""Run 100k closed M1 bar comparison on M15 and M5 for BTCUSDT and SOLUSDT."""
from __future__ import annotations
import json, math, time
from pathlib import Path
import pandas as pd
from pragma.data.binance_data import BinanceDataClient
from pragma.backtest_engine import simulate_walk_forward
from scripts.run_real_comparison import aggregate

def main():
    reports = []
    out = Path("artifacts/real_walk_forward_100k_comparison.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with BinanceDataClient(cache_dir="data_cache") as client:
        for symbol in ["BTCUSDT", "SOLUSDT"]:
            m1 = client.rates(symbol, "M1", count=100000)
            for tf, minutes in [("M15", 15), ("M5", 5)]:
                frame = aggregate(m1, minutes)
                if "tick_volume" not in frame and "volume" in frame:
                    frame["tick_volume"] = frame["volume"]
                for engine in ["amt", "orderflow", "hybrid"]:
                    strict = engine in ("orderflow", "hybrid")
                    t0 = time.time()
                    r = simulate_walk_forward(frame, symbol=symbol, engine=engine, strict_real_delta=strict)
                    dt = time.time() - t0
                    r["timeframe"] = tf
                    r["bars_downloaded_m1"] = len(m1)
                    reports.append(r)
                    out.write_text(json.dumps(reports, indent=2, allow_nan=True, default=str) + "\n", encoding="utf-8")
                    pf_str = "inf" if math.isinf(r["profit_factor"]) else f"{r['profit_factor']:.2f}"
                    print(f"[{symbol} {tf} {engine}] {r['total_trades']} trades, Net R: {r['net_r']:+.2f}, WR: {r['win_rate']*100:.1f}%, PF: {pf_str} ({dt:.1f}s)", flush=True)

    print("\n### 100.000 Barlık Gerçek Binance Karşılaştırma Sonuç Tablosu\n")
    print("| Symbol | Timeframe | Engine | Bars | Trades | Win Rate | Profit Factor | Net R | Exp R | Max DD R | t-stat | Absorption (Tr/WR) | Exhaustion (Tr/WR) |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in reports:
        by_div = r.get("by_divergence_type", {})
        abs_info = by_div.get("ABSORPTION", {})
        exh_info = by_div.get("EXHAUSTION", {})
        abs_str = f"{abs_info.get('trades', 0)} ({abs_info.get('win_rate', 0.0)*100:.1f}%)" if abs_info.get('trades', 0) > 0 else "-"
        exh_str = f"{exh_info.get('trades', 0)} ({exh_info.get('win_rate', 0.0)*100:.1f}%)" if exh_info.get('trades', 0) > 0 else "-"
        pf_str = "inf" if math.isinf(r['profit_factor']) else f"{r['profit_factor']:.2f}"
        print(f"| {r['symbol']} | {r.get('timeframe','-')} | {r['engine']} | {r.get('bars_processed',0)} | {r['total_trades']} | {r['win_rate']*100:.1f}% | {pf_str} | {r['net_r']:+.2f}R | {r['expectancy_r']:+.2f}R | {r['max_drawdown_r']:.2f}R | {r['t_stat']:.2f} | {abs_str} | {exh_str} |")

if __name__ == "__main__":
    main()
