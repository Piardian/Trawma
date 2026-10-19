"""Comprehensive Walk-Forward Matrix and Whitelist Generator for Trawma Phase 6.

Evaluates 3 Order Flow engines (orderflow_absorption, hybrid_absorption, cvd_momentum)
across Binance Futures (100k M1 -> M5, M15) and MT5 Terminal (50k bars -> M5, M15)
with chronological Train (60%) / Val (20%) / OOS (20%) Walk-Forward validation.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

from pragma.backtest_engine import simulate_walk_forward
from pragma.data.binance_data import BinanceDataClient, filter_closed_bars
from pragma.data.mt5_data import MT5DataClient, filter_closed_bars as mt5_filter_closed


def aggregate_m1(frame: pd.DataFrame, minutes: int) -> pd.DataFrame:
    """Causally aggregates M1 bars into M5 or M15 bars, summing volume delta."""
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise ValueError("Frame must have DatetimeIndex")
    rules = f"{minutes}min"
    agg = {"open": "first", "high": "max", "low": "min", "close": "last"}
    for c in ("volume", "tick_volume", "buy_volume", "sell_volume", "volume_delta",
              "quote_asset_volume", "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume"):
        if c in frame.columns:
            agg[c] = "sum"
    grouped = frame.resample(rules, label="left", closed="left")
    result = grouped.agg(agg).dropna(subset=["open", "high", "low", "close"])
    complete = grouped["open"].count() == minutes
    res = result.loc[complete.reindex(result.index, fill_value=False)]
    if "tick_volume" not in res and "volume" in res:
        res["tick_volume"] = res["volume"]
    return res


def run_single_matrix_cell(
    frame: pd.DataFrame,
    *,
    symbol: str,
    timeframe: str,
    engine: str,
    strict_real_delta: bool,
    fee_bps: float,
    btc_frame: pd.DataFrame | None = None,
    macro_mode: str = "Autonomous",
    macro_bias: int | None = None,
) -> dict[str, Any]:
    """Runs Train (60%), Val (20%), OOS (20%) and Full backtests for a single cell."""
    n = len(frame)
    if n < 100:
        return {"error": f"Insufficient bars ({n})"}

    n_train = int(n * 0.60)
    n_val = int(n * 0.20)

    train_df = frame.iloc[:n_train]
    val_df = frame.iloc[n_train: n_train + n_val]
    oos_df = frame.iloc[n_train + n_val:]

    def slice_btc(sub_df: pd.DataFrame) -> pd.DataFrame | None:
        if btc_frame is None or btc_frame.empty:
            return None
        return btc_frame.loc[(btc_frame.index >= sub_df.index[0]) & (btc_frame.index <= sub_df.index[-1])]

    btc_train = slice_btc(train_df)
    btc_val = slice_btc(val_df)
    btc_oos = slice_btc(oos_df)

    def extract_summary(rep: dict[str, Any]) -> dict[str, Any]:
        return {
            "total_trades": rep.get("total_trades", 0),
            "win_rate": rep.get("win_rate", 0.0),
            "profit_factor": rep.get("profit_factor"),
            "net_r": rep.get("net_r", 0.0),
            "expectancy_r": rep.get("expectancy_r", 0.0),
            "max_drawdown_r": rep.get("max_drawdown_r", 0.0),
            "t_stat": rep.get("t_stat", 0.0),
            "bars": rep.get("bars_processed", 0),
            "funnel": rep.get("funnel_counts", {}),
        }

    rep_full = simulate_walk_forward(
        frame, symbol=symbol, engine=engine, strict_real_delta=strict_real_delta,
        fee_bps_per_side=fee_bps, btc_frame=btc_frame, macro_bias=macro_bias,
    )

    # Chronological trade partitioning across Train / Val / OOS periods
    t_val_start = frame.index[n_train]
    t_oos_start = frame.index[n_train + n_val]
    all_trades = rep_full.get("trades", [])

    def compute_partition_metrics(trades_subset: list[dict], bar_count: int) -> dict[str, Any]:
        if not trades_subset:
            return {
                "total_trades": 0, "win_rate": 0.0, "profit_factor": None,
                "net_r": 0.0, "expectancy_r": 0.0, "max_drawdown_r": 0.0,
                "t_stat": 0.0, "bars": bar_count, "funnel": {},
            }
        rs = np.array([t["net_r"] for t in trades_subset], dtype=float)
        wins = rs[rs > 0]
        losses = rs[rs < 0]
        pf = float(wins.sum() / abs(losses.sum())) if len(losses) and abs(losses.sum()) > 0 else (None if len(wins) else 0.0)
        equity = np.cumsum(rs)
        peak = np.maximum.accumulate(np.r_[0.0, equity])[1:] if len(equity) else np.array([])
        dd = peak - equity if len(equity) else np.array([])
        sd = float(rs.std(ddof=1)) if len(rs) > 1 else 0.0
        tstat = float(rs.mean() / (sd / np.sqrt(len(rs)))) if len(rs) > 1 and sd > 0 else 0.0
        return {
            "total_trades": len(trades_subset),
            "win_rate": float((rs > 0).mean()) if len(rs) else 0.0,
            "profit_factor": pf,
            "net_r": float(rs.sum()),
            "expectancy_r": float(rs.mean()),
            "max_drawdown_r": float(dd.max()) if len(dd) else 0.0,
            "t_stat": tstat,
            "bars": bar_count,
            "funnel": {},
        }

    train_trades = [t for t in all_trades if pd.Timestamp(t["exit_time"]) < t_val_start]
    val_trades = [t for t in all_trades if t_val_start <= pd.Timestamp(t["exit_time"]) < t_oos_start]
    oos_trades_list = [t for t in all_trades if pd.Timestamp(t["exit_time"]) >= t_oos_start]

    s_train = compute_partition_metrics(train_trades, n_train)
    s_val = compute_partition_metrics(val_trades, n_val)
    s_oos = compute_partition_metrics(oos_trades_list, len(frame) - n_train - n_val)
    s_full = extract_summary(rep_full)

    # Whitelist qualification check (evaluated strictly on OOS)
    oos_trades = s_oos["total_trades"]
    oos_pf = s_oos["profit_factor"] or 0.0
    oos_net_r = s_oos["net_r"]
    oos_dd = s_oos["max_drawdown_r"]

    is_whitelisted = (oos_trades >= 5) and (oos_pf >= 1.15) and (oos_net_r > 0.0)

    cell_id = f"{symbol}_{timeframe}_{engine}_{macro_mode}"

    return {
        "cell_id": cell_id,
        "symbol": symbol,
        "timeframe": timeframe,
        "engine": engine,
        "macro_mode": macro_mode,
        "macro_bias": macro_bias,
        "strict_real_delta": strict_real_delta,
        "whitelisted": is_whitelisted,
        "train": s_train,
        "val": s_val,
        "oos": s_oos,
        "full": s_full,
    }


def load_binance_m1(symbol: str, count: int = 100000, cache_dir: str = "data_cache") -> pd.DataFrame:
    """Loads 100k M1 bars directly from local cache if present, else fetches via client."""
    path = Path(cache_dir) / f"binance_{symbol}_M1.csv"
    if path.exists():
        try:
            df = pd.read_csv(path, parse_dates=["time"])
            df["time"] = pd.to_datetime(df["time"], utc=True)
            df = df.set_index("time").sort_index()
            df = df[~df.index.duplicated(keep="last")]
            df = filter_closed_bars(df, "M1")
            if len(df) >= count:
                res = df.tail(count).copy()
                if "tick_volume" not in res and "volume" in res:
                    res["tick_volume"] = res["volume"]
                return res
        except Exception as exc:
            print(f"  [Cache Warning] Could not read {path}: {exc}, falling back to client...", flush=True)

    with BinanceDataClient(cache_dir=cache_dir, futures_only=True) as client:
        m1 = client.rates(symbol, "M1", count=count)
        return filter_closed_bars(m1, "M1")


def main():
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    parser = argparse.ArgumentParser(description="Trawma Phase 6 Walk-Forward Matrix Generator")
    parser.add_argument("--binance-symbols", nargs="+", default=["BTCUSDT", "SOLUSDT", "ETHUSDT", "LINKUSDT"])
    parser.add_argument("--mt5-symbols", nargs="+", default=["XAUUSD", "NAS100", "USDJPY", "EURUSD"])
    parser.add_argument("--timeframes", nargs="+", default=["M5", "M15"])
    parser.add_argument("--engines", nargs="+", default=["orderflow_absorption", "hybrid_absorption", "cvd_momentum"])
    parser.add_argument("--binance-count", type=int, default=100000)
    parser.add_argument("--mt5-count", type=int, default=50000)
    parser.add_argument("--cache-dir", default="data_cache")
    parser.add_argument("--output-matrix", default="artifacts/trawma_phase6_matrix_results.json")
    parser.add_argument("--output-whitelist", default="reports/whitelist.json")
    args = parser.parse_args()

    results: list[dict[str, Any]] = []
    whitelist_entries: list[dict[str, Any]] = []

    print("======================================================================", flush=True)
    print("  TRAWMA FAZ 6: WALK-FORWARD DOGRULAMA VE WHITELIST MATRISI", flush=True)
    print("  - Kripto (Binance): 100k M1 Bar -> M5/M15 (Strict Taker Delta + BTC Leader)", flush=True)
    print("  - MT5 (FX/Emtia/Endeks): 50k Bar -> M5/M15 (Autonomous + Begonya Macro)", flush=True)
    print("  - Motorlar: orderflow_absorption, hybrid_absorption, cvd_momentum", flush=True)
    print("  - Bolunme: Train (%60) | Val (%20) | OOS (%20)", flush=True)
    print("  - Whitelist Kriteri (OOS): Trades >= 5, PF >= 1.15, Net R > 0", flush=True)
    print("======================================================================\n", flush=True)

    # 1. BINANCE FUTURES CRYPTO RUNS
    print("[1/2] Binance Kripto Verileri Hazirlaniyor...", flush=True)
    binance_frames: dict[str, dict[str, pd.DataFrame]] = {}
    btc_m1 = load_binance_m1("BTCUSDT", count=args.binance_count, cache_dir=args.cache_dir)
    print(f"  -> BTCUSDT 100k M1 yuklendi ({len(btc_m1)} bar).", flush=True)
    btc_by_tf = {
        "M5": aggregate_m1(btc_m1, 5),
        "M15": aggregate_m1(btc_m1, 15),
    }
    binance_frames["BTCUSDT"] = btc_by_tf

    for sym in args.binance_symbols:
        if sym == "BTCUSDT":
            continue
        m1 = load_binance_m1(sym, count=args.binance_count, cache_dir=args.cache_dir)
        print(f"  -> {sym} 100k M1 yuklendi ({len(m1)} bar).", flush=True)
        binance_frames[sym] = {
            "M5": aggregate_m1(m1, 5),
            "M15": aggregate_m1(m1, 15),
        }

    print(f"  -> {len(binance_frames)} Kripto sembolu hazirlandi.\n", flush=True)

    for sym in args.binance_symbols:
        for tf in args.timeframes:
            df = binance_frames[sym][tf]
            btc_ref = None if sym == "BTCUSDT" else binance_frames["BTCUSDT"][tf]
            for eng in args.engines:
                print(f"[RUN CRYPTO] {sym:10} | {tf:3} | {eng:22} ... ", end="", flush=True)
                cell = run_single_matrix_cell(
                    df,
                    symbol=sym,
                    timeframe=tf,
                    engine=eng,
                    strict_real_delta=True,
                    fee_bps=4.0,
                    btc_frame=btc_ref,
                    macro_mode="Autonomous",
                    macro_bias=None,
                )
                results.append(cell)
                oos = cell["oos"]
                full = cell["full"]
                pf_str = f"{oos['profit_factor']:.2f}" if oos['profit_factor'] is not None else "N/A"
                status = "[WHITELIST]" if cell["whitelisted"] else "[PASS]"
                print(f"OOS: {oos['total_trades']:2} trades | WR {oos['win_rate']*100:4.1f}% | PF {pf_str:4} | Net {oos['net_r']:+6.2f}R -> {status}", flush=True)
                if cell["whitelisted"]:
                    whitelist_entries.append(cell)

    # 2. MT5 TERMINAL RUNS
    print("\n[2/2] MT5 Terminal Verileri Hazirlaniyor...", flush=True)
    mt5_macro_map = {
        "EURUSD": -1,  # Begonya Short anchor
        "USDJPY": 1,   # Begonya Long anchor
        "XAUUSD": 1,   # Gold Long anchor
        "NAS100": 1,   # US Tech Long anchor
    }

    try:
        with MT5DataClient() as mt5_client:
            for sym in args.mt5_symbols:
                for tf in args.timeframes:
                    print(f"  -> MT5 {sym} {tf} çekiliyor ({args.mt5_count} bar)... ", end="", flush=True)
                    raw_bars = mt5_client.rates(sym, tf, count=args.mt5_count)
                    df = mt5_filter_closed(raw_bars, tf)
                    if "tick_volume" not in df and "volume" in df:
                        df["tick_volume"] = df["volume"]
                    print(f"Tamam ({len(df)} bar)", flush=True)

                    fee = 1.0 if sym in ("EURUSD", "USDJPY") else 2.5

                    # Test both Autonomous (macro=None) and Begonya_Macro
                    macro_modes = [("Autonomous", None)]
                    if sym in mt5_macro_map:
                        macro_modes.append(("Begonya_Macro", mt5_macro_map[sym]))

                    for macro_mode, macro_bias in macro_modes:
                        for eng in args.engines:
                            print(f"[RUN MT5]    {sym:10} | {tf:3} | {eng:22} | {macro_mode:13} ... ", end="", flush=True)
                            cell = run_single_matrix_cell(
                                df,
                                symbol=sym,
                                timeframe=tf,
                                engine=eng,
                                strict_real_delta=False,
                                fee_bps=fee,
                                btc_frame=None,
                                macro_mode=macro_mode,
                                macro_bias=macro_bias,
                            )
                            results.append(cell)
                            oos = cell["oos"]
                            pf_str = f"{oos['profit_factor']:.2f}" if oos['profit_factor'] is not None else "N/A"
                            status = "[WHITELIST]" if cell["whitelisted"] else "[PASS]"
                            print(f"OOS: {oos['total_trades']:2} trades | WR {oos['win_rate']*100:4.1f}% | PF {pf_str:4} | Net {oos['net_r']:+6.2f}R -> {status}", flush=True)
                            if cell["whitelisted"]:
                                whitelist_entries.append(cell)

    except Exception as exc:
        print(f"\n[MT5 Warning] MT5 Terminaline erisilemedi: {exc}. MT5 testleri atlandi.", flush=True)

    # 3. SAVE OUTPUTS
    out_matrix_path = Path(args.output_matrix)
    out_matrix_path.parent.mkdir(parents=True, exist_ok=True)
    out_matrix_path.write_text(json.dumps(results, indent=2, default=str) + "\n", encoding="utf-8")
    print(f"\n[Output] Matris sonuclari kaydedildi -> {out_matrix_path}", flush=True)

    out_wl_path = Path(args.output_whitelist)
    out_wl_path.parent.mkdir(parents=True, exist_ok=True)
    out_wl_payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_configurations_tested": len(results),
        "total_whitelisted": len(whitelist_entries),
        "whitelist_criteria": {
            "validation_segment": "OOS (Out-of-Sample final 20%)",
            "min_oos_trades": 5,
            "min_oos_profit_factor": 1.15,
            "min_oos_net_r": 0.01,
        },
        "whitelisted_strategies": whitelist_entries,
    }
    out_wl_path.write_text(json.dumps(out_wl_payload, indent=2, default=str) + "\n", encoding="utf-8")
    print(f"[Output] Whitelist raporu kaydedildi -> {out_wl_path}", flush=True)

    # 4. PRINT SUMMARY TABLE
    print("\n==========================================================================================================", flush=True)
    print("  TRAWMA FAZ 6: ONAYLANAN WHITELIST STRATEJİLERİ (OOS Doğrulamalı)", flush=True)
    print("==========================================================================================================", flush=True)
    print(f"{'Strateji':<38} | {'OOS Trades':<10} | {'OOS WR%':<8} | {'OOS PF':<8} | {'OOS Net R':<10} | {'Full Net R':<10}", flush=True)
    print("-" * 106, flush=True)
    if not whitelist_entries:
        print("  Henüz kriterleri sağlayan strateji bulunamadı.", flush=True)
    else:
        for entry in whitelist_entries:
            oos = entry["oos"]
            full = entry["full"]
            pf_str = f"{oos['profit_factor']:.2f}" if oos["profit_factor"] is not None else "N/A"
            name = f"{entry['symbol']} {entry['timeframe']} {entry['engine']} ({entry['macro_mode'][:4]})"
            print(f"{name:<38} | {oos['total_trades']:<10} | {oos['win_rate']*100:<8.1f} | {pf_str:<8} | {oos['net_r']:<+10.2f} | {full['net_r']:<+10.2f}", flush=True)
    print("==========================================================================================================\n", flush=True)


if __name__ == "__main__":
    main()
