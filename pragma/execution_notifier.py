"""Execution notifier and deduplicator for Trawma live signals.

Formats Order Flow absorption signals into structured alerts, prevents duplicate
signals for the same bar, and sends notifications via Telegram Bot API and console.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime, timezone
import json
import os
from typing import Any, Mapping
import urllib.parse
import urllib.request

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass


class SignalDeduplicator:
    """Thread-safe in-memory deduplicator for executed bar signals."""

    def __init__(self, max_history: int = 1000):
        self._seen: set[tuple[str, str, str]] = set()
        self._order: list[tuple[str, str, str]] = []
        self._max_history = max(10, max_history)

    def is_new(self, symbol: str, timeframe: str, bar_time: str | datetime) -> bool:
        time_key = str(bar_time)
        key = (symbol.upper(), timeframe.upper(), time_key)
        if key in self._seen:
            return False
        self._seen.add(key)
        self._order.append(key)
        if len(self._order) > self._max_history:
            old = self._order.pop(0)
            self._seen.discard(old)
        return True

    def clear(self) -> None:
        self._seen.clear()
        self._order.clear()


@dataclass
class OrderFlowSignal:
    symbol: str
    timeframe: str
    side: str  # "LONG" or "SHORT"
    entry_price: float
    stop_loss: float
    tp1: float
    tp2: float
    risk: float
    bar_time: str
    poi_info: Mapping[str, Any]
    delta_metrics: Mapping[str, Any]
    divergence_type: str = "ABSORPTION"
    macro_bias: str = "UNKNOWN"
    engine: str = "TRAWMA_ORDERFLOW"


def format_signal_text(sig: OrderFlowSignal, format_type: str = "text", scoreboard: str | None = None) -> str:
    side_icon = "🟢 LONG" if sig.side == "LONG" else "🔴 SHORT"
    rr_tp1 = abs(sig.tp1 - sig.entry_price) / max(1e-9, sig.risk)
    rr_tp2 = abs(sig.tp2 - sig.entry_price) / max(1e-9, sig.risk)

    poi_str = "N/A"
    if sig.poi_info:
        kind = sig.poi_info.get("kind") or sig.poi_info.get("type", "POI")
        level = sig.poi_info.get("level", 0.0)
        poi_str = f"{kind} @ {level:.5f}"

    b_delta = sig.delta_metrics.get("bar_delta", 0.0)
    cvd_val = sig.delta_metrics.get("cvd_value", 0.0)
    z_score = sig.delta_metrics.get("delta_zscore", 0.0)

    if format_type == "telegram_html":
        side_icon = "🟢 LONG" if sig.side == "LONG" else "🔴 SHORT"
        msg = (
            f"<b>⚡ TRAWMA ORDER FLOW SİNYALİ</b>\n\n"
            f"<b>Sembol:</b> <code>{sig.symbol}</code> ({sig.timeframe})\n"
            f"<b>Yön:</b> {side_icon}\n"
            f"<b>Motor:</b> <code>{sig.engine} (Absorption-Only)</code>\n"
            f"<b>Makro Kapı (Begonya):</b> <code>{sig.macro_bias}</code>\n"
            f"<b>POI Seviyesi:</b> {poi_str}\n"
            f"<b>CVD Uyumsuzluk:</b> <b>{sig.divergence_type}</b>\n"
            f"<b>Delta İstatistiği:</b> Delta: {b_delta:+.2f} | CVD: {cvd_val:+.2f} | Z: {z_score:+.2f}\n\n"
            f"<b>Giriş (Entry):</b> <code>{sig.entry_price:.5f}</code>\n"
            f"<b>Stop Loss (SL):</b> <code>{sig.stop_loss:.5f}</code> (Risk: {sig.risk:.5f})\n"
            f"<b>TP1 (1R):</b> <code>{sig.tp1:.5f}</code> (R:R {rr_tp1:.1f}R)\n"
            f"<b>TP2 (2R):</b> <code>{sig.tp2:.5f}</code> (R:R {rr_tp2:.1f}R)\n"
            f"<b>Bar Zamanı (UTC):</b> <code>{sig.bar_time}</code>"
        )
        if scoreboard:
            msg += f"\n\n{scoreboard}"
        return msg

    # Console plain text format
    side_label = "[LONG]" if sig.side == "LONG" else "[SHORT]"
    out = (
        f"\n=======================================================\n"
        f"[TRAWMA ORDER FLOW CANLI SINYAL]\n"
        f"Sembol       : {sig.symbol} ({sig.timeframe})\n"
        f"Yon          : {side_label}\n"
        f"Motor        : {sig.engine} (Absorption-Only)\n"
        f"Begonya Makro: {sig.macro_bias}\n"
        f"POI Havuzu   : {poi_str}\n"
        f"Divergence   : {sig.divergence_type}\n"
        f"Delta Durumu : Delta={b_delta:+.2f}, CVD={cvd_val:+.2f}, Z={z_score:+.2f}\n"
        f"-------------------------------------------------------\n"
        f"Giris        : {sig.entry_price:.5f}\n"
        f"Stop Loss    : {sig.stop_loss:.5f} (Risk: {sig.risk:.5f})\n"
        f"TP1          : {sig.tp1:.5f} (R:R {rr_tp1:.1f}R)\n"
        f"TP2          : {sig.tp2:.5f} (R:R {rr_tp2:.1f}R)\n"
        f"Bar Zamani   : {sig.bar_time} UTC\n"
        f"=======================================================\n"
    )
    if scoreboard:
        out += f"\n{scoreboard}\n"
    return out


def send_telegram_alert(message_html: str, bot_token: str | None = None, chat_id: str | None = None) -> bool:
    token = bot_token or os.getenv("TELEGRAM_BOT_TOKEN")
    chat = chat_id or os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat:
        return False

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        "chat_id": chat,
        "text": message_html,
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
    }
    data = urllib.parse.urlencode(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status == 200
    except Exception as exc:
        print(f"[Notifier Warning] Telegram bildirim hatası: {exc}", flush=True)
        return False


class ExecutionNotifier:
    """Manages signal deduplication and dispatch across Console & Telegram."""

    def __init__(self, bot_token: str | None = None, chat_id: str | None = None):
        self.deduplicator = SignalDeduplicator()
        self.bot_token = bot_token
        self.chat_id = chat_id

    def notify(self, signal: OrderFlowSignal, scoreboard: str | None = None) -> bool:
        """Deduplicate and notify. Returns True if signal was newly dispatched."""
        if not self.deduplicator.is_new(signal.symbol, signal.timeframe, signal.bar_time):
            return False

        # 1. Console print
        console_msg = format_signal_text(signal, format_type="text", scoreboard=scoreboard)
        print(console_msg, flush=True)

        # 2. Telegram message
        tg_html = format_signal_text(signal, format_type="telegram_html", scoreboard=scoreboard)
        sent = send_telegram_alert(tg_html, self.bot_token, self.chat_id)
        if sent:
            print(f"[Notifier] Telegram bildirimi iletildi -> {signal.symbol} {signal.side}")
        return True
