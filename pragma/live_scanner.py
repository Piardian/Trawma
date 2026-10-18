"""Live sentinel scanner for Trawma order flow signals.

Supports:
- MT5 Whitelist scanning with Begonya macro gate alignment (--scan-begonya)
- Binance Futures crypto scanning with BTC CVD leadership (--scan-crypto)
- Quiet execution (no repeated spam when no new bar has closed)
- Deduplicated notifications via Console and Telegram Bot API
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

import pandas as pd

from .instrument_profiles import (
    WHITELIST_INSTRUMENTS,
    BLACKLIST_FOREX_SYMBOLS,
    is_whitelisted,
    is_blacklisted,
    get_profile,
)
from .macro_gate import read_macro_bias, is_macro_aligned, find_macro_bias_file
from .execution_notifier import ExecutionNotifier, OrderFlowSignal
from .orderflow_engine import (
    OrderFlowConfig,
    calculate_orderflow_features,
    detect_orderflow_pools,
    evaluate_orderflow_trigger,
)
from .backtest_engine import btc_cvd_regime_filter


class LiveSentinelScanner:
    """Orchestrates live scanning, macro gating, and signal emission."""

    def __init__(
        self,
        notifier: ExecutionNotifier | None = None,
        macro_path: Path | str | None = None,
        config: OrderFlowConfig | None = None,
    ):
        self.notifier = notifier or ExecutionNotifier()
        self.macro_path = macro_path or find_macro_bias_file()
        self.config = config or OrderFlowConfig()
        self._last_processed_bar_time: dict[tuple[str, str], pd.Timestamp] = {}

    def scan_mt5_symbol(self, client, symbol: str, macro_required: bool = True) -> OrderFlowSignal | None:
        """Scan a single MT5 instrument (e.g. XAUUSD)."""
        if is_blacklisted(symbol):
            return None
        profile = get_profile(symbol)
        tf = profile.optimal_timeframe if profile else "M5"

        from .data.mt5_data import filter_closed_bars

        # 1. Fetch recent bars (e.g. 150 bars is enough for ATR, VWAP and recent pools)
        raw_bars = client.rates(symbol, tf, count=150)
        closed = filter_closed_bars(raw_bars, tf)
        if closed.empty or len(closed) < 50:
            return None

        last_bar_time = closed.index[-1]
        key = (symbol.upper(), tf.upper())

        # 2. Quiet check: do nothing if no new bar has closed
        if self._last_processed_bar_time.get(key) == last_bar_time:
            return None

        self._last_processed_bar_time[key] = last_bar_time

        # 3. Check Begonya Macro Gate
        macro_decision = read_macro_bias(symbol, path=self.macro_path)
        if macro_required:
            if not macro_decision.is_executable:
                # Macro says NEUTRAL_RANGE, EVENT_FREEZE, or capital preservation
                return None

        # 4. Compute order flow features
        features = calculate_orderflow_features(closed, config=self.config)
        pools = detect_orderflow_pools(features, config=self.config)
        last_bar = features.iloc[-1]
        atr = float(last_bar.atr_14) if "atr_14" in last_bar else 0.0
        if atr <= 0:
            return None

        touched = [p for p in pools if float(last_bar.low) <= p.upper and float(last_bar.high) >= p.lower]
        if not touched:
            return None

        pool = next((p for p in reversed(touched) if p.side in ("LONG", "SHORT")), None)
        if pool is None:
            return None

        side = pool.side

        # Macro alignment check
        if macro_required and macro_decision.allowed_side != side:
            return None

        # 5. Evaluate trigger
        decision = evaluate_orderflow_trigger(
            features,
            side,
            poi=pool,
            config=self.config,
            strict_real_delta=False,
        )

        if decision.action != "EXECUTE":
            return None

        entry_price = float(last_bar.close)
        risk = max(atr * 1.0, entry_price * 0.0001 * 2.5)
        stop_loss = entry_price - risk if side == "LONG" else entry_price + risk
        tp1 = entry_price + 1.0 * risk * (1 if side == "LONG" else -1)
        tp2 = entry_price + 2.0 * risk * (1 if side == "LONG" else -1)

        sig = OrderFlowSignal(
            symbol=symbol,
            timeframe=tf,
            side=side,
            entry_price=entry_price,
            stop_loss=stop_loss,
            tp1=tp1,
            tp2=tp2,
            risk=risk,
            bar_time=str(last_bar_time),
            poi_info={"kind": pool.kind, "level": pool.level, "lower": pool.lower, "upper": pool.upper},
            delta_metrics=decision.metrics,
            divergence_type=str(decision.metrics.get("cvd_divergence_type", "ABSORPTION")),
            macro_bias=macro_decision.raw_gate,
            engine="TRAWMA_ORDERFLOW",
        )

        self.notifier.notify(sig)
        return sig

    def scan_crypto_symbol(self, client, symbol: str, btc_frame: pd.DataFrame | None = None) -> OrderFlowSignal | None:
        """Scan a single Binance Futures crypto instrument (e.g. BTCUSDT, SOLUSDT)."""
        profile = get_profile(symbol)
        tf = profile.optimal_timeframe if profile else "M15"
        minutes = 15 if tf == "M15" else 5

        from .data.binance_data import filter_closed_bars
        from scripts.run_real_comparison import aggregate

        # 1. Fetch recent M1 bars and aggregate
        m1 = client.rates(symbol, "M1", count=150 * minutes)
        m1 = filter_closed_bars(m1, "M1")
        if m1.empty or len(m1) < 50 * minutes:
            return None

        frame = aggregate(m1, minutes)
        if frame.empty or len(frame) < 50:
            return None

        if "tick_volume" not in frame and "volume" in frame:
            frame["tick_volume"] = frame["volume"]

        last_bar_time = frame.index[-1]
        key = (symbol.upper(), tf.upper())

        # 2. Quiet check: do nothing if no new bar has closed
        if self._last_processed_bar_time.get(key) == last_bar_time:
            return None

        self._last_processed_bar_time[key] = last_bar_time

        # 3. Altcoin BTC CVD leadership check
        if symbol.upper() != "BTCUSDT" and btc_frame is not None:
            btc_regimes = btc_cvd_regime_filter(btc_frame, config=self.config)
            eligible = btc_regimes.loc[btc_regimes.index <= last_bar_time]
            regime = str(eligible.iloc[-1]) if not eligible.empty else "NEUTRAL"
            if regime == "NEUTRAL":
                return None
        else:
            regime = "LEADER_DIRECT"

        # 4. Features & POIs
        features = calculate_orderflow_features(frame, config=self.config)
        pools = detect_orderflow_pools(features, config=self.config)
        last_bar = features.iloc[-1]
        atr = float(last_bar.atr_14) if "atr_14" in last_bar else 0.0
        if atr <= 0:
            return None

        touched = [p for p in pools if float(last_bar.low) <= p.upper and float(last_bar.high) >= p.lower]
        if not touched:
            return None

        pool = next((p for p in reversed(touched) if p.side in ("LONG", "SHORT")), None)
        if pool is None:
            return None

        side = pool.side

        # If altcoin, ensure BTC CVD regime matches signal side
        if symbol.upper() != "BTCUSDT":
            if side == "LONG" and regime != "IMBALANCE_BULL":
                return None
            if side == "SHORT" and regime != "IMBALANCE_BEAR":
                return None

        # 5. Evaluate trigger
        decision = evaluate_orderflow_trigger(
            features,
            side,
            poi=pool,
            config=self.config,
            strict_real_delta=True,
        )

        if decision.action != "EXECUTE":
            return None

        entry_price = float(last_bar.close)
        risk = max(atr * 1.0, entry_price * 0.0004 * 2.5)
        stop_loss = entry_price - risk if side == "LONG" else entry_price + risk
        tp1 = entry_price + 1.0 * risk * (1 if side == "LONG" else -1)
        tp2 = entry_price + 2.0 * risk * (1 if side == "LONG" else -1)

        sig = OrderFlowSignal(
            symbol=symbol,
            timeframe=tf,
            side=side,
            entry_price=entry_price,
            stop_loss=stop_loss,
            tp1=tp1,
            tp2=tp2,
            risk=risk,
            bar_time=str(last_bar_time),
            poi_info={"kind": pool.kind, "level": pool.level, "lower": pool.lower, "upper": pool.upper},
            delta_metrics=decision.metrics,
            divergence_type=str(decision.metrics.get("cvd_divergence_type", "ABSORPTION")),
            macro_bias=f"BTC_REGIME_{regime}",
            engine="TRAWMA_ORDERFLOW",
        )

        self.notifier.notify(sig)
        return sig


def run_sentinel_pipeline(
    scan_begonya: bool = False,
    scan_crypto: bool = False,
    watch: bool = False,
    interval_sec: int = 15,
    refresh_macro: bool = True,
    macro_path: Path | str | None = None,
) -> int:
    """Main CLI entrypoint for Trawma sentinel watch mode."""
    scanner = LiveSentinelScanner(macro_path=macro_path)

    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    print("\n=======================================================")
    print("[TRAWMA LIVE SENTINEL WATCHER ACTIVE]")
    print("=======================================================")
    if scan_begonya:
        print("  - MT5 Begonya Taramasi : AKTIF (XAUUSD M5 Whitelist)")
    if scan_crypto:
        print("  - Binance Kripto       : AKTIF (BTCUSDT M15, SOLUSDT M15)")
    print(f"  - Surekli Izleme (--watch): {'ACIK' if watch else 'KAPALI (Tek Gecis)'}")
    print(f"  - Tarama Periyodu      : {interval_sec} saniye")
    macro_file = find_macro_bias_file([Path(macro_path)] if macro_path else None)
    print(f"  - Begonya Makro Dosyasi: {macro_file or 'Bulunamadi'}")
    print("=======================================================\n", flush=True)

    iteration = 0
    try:
        while True:
            iteration += 1
            now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

            # 1. Begonya / MT5 scan
            if scan_begonya:
                try:
                    from .data.mt5_data import MT5DataClient
                    with MT5DataClient() as mt5_client:
                        # Scan Whitelisted MT5 symbols (e.g. XAUUSD)
                        for sym in ["XAUUSD"]:
                            scanner.scan_mt5_symbol(mt5_client, sym, macro_required=True)
                except Exception as exc:
                    print(f"[{now_str}] MT5 Tarama Hatası: {exc}", flush=True)

            # 2. Crypto scan
            if scan_crypto:
                try:
                    from .data.binance_data import BinanceDataClient, filter_closed_bars
                    from scripts.run_real_comparison import aggregate
                    with BinanceDataClient(futures_only=True) as bnc_client:
                        # First get BTC reference frame for leadership
                        btc_m1 = bnc_client.rates("BTCUSDT", "M1", count=150 * 15)
                        btc_m1 = filter_closed_bars(btc_m1, "M1")
                        btc_m15 = aggregate(btc_m1, 15) if not btc_m1.empty else None

                        scanner.scan_crypto_symbol(bnc_client, "BTCUSDT", btc_frame=None)
                        scanner.scan_crypto_symbol(bnc_client, "SOLUSDT", btc_frame=btc_m15)
                except Exception as exc:
                    print(f"[{now_str}] Binance Tarama Hatası: {exc}", flush=True)

            if not watch:
                print(f"[{now_str}] Tek geçişli tarama tamamlandı.")
                break

            time.sleep(max(1, interval_sec))

    except KeyboardInterrupt:
        print("\n[Sentinel] Canlı nöbetçi modu kullanıcı tarafından sonlandırıldı.")
        return 0

    return 0
