"""Execution Bridge for Trawma live/paper trading and order management.

Supports three execution modes:
- notify_only: Dispatch alerts to Telegram & Console without opening orders.
- paper: Live tracking of virtual positions in artifacts/paper_positions.json,
  managing 50% partial TP1, Breakeven SL modification, TP2, 16-bar Time-Stop,
  and Friday Weekend Close with Telegram notifications.
- mt5_live: Safety-gated real execution via MT5 terminal with automatic lot sizing
  based on risk percentage (default 0.5%), partial closes, and magic number 202604.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping

from .execution_notifier import ExecutionNotifier, OrderFlowSignal, send_telegram_alert


TRAWMA_MAGIC_NUMBER = 202604
DEFAULT_PAPER_PATH = Path("artifacts/paper_positions.json")


def calculate_mt5_lot_size(
    balance: float,
    entry_price: float,
    stop_loss: float,
    symbol_info: Any,
    risk_pct: float = 0.5,
) -> float:
    """Calculate exact MT5 lot size based on account balance and risk percentage."""
    if balance <= 0:
        raise ValueError("balance must be positive")
    if risk_pct <= 0 or risk_pct > 10.0:
        raise ValueError("risk_pct must be between 0.01% and 10.0%")

    risk_amount = balance * (risk_pct / 100.0)
    stop_distance = abs(entry_price - stop_loss)
    if stop_distance <= 0:
        raise ValueError("stop distance must be greater than zero")

    # In MT5, trade_tick_value is profit for 1 lot when price moves by trade_tick_size
    tick_size = float(getattr(symbol_info, "trade_tick_size", 0.0) or getattr(symbol_info, "point", 0.00001))
    tick_value = float(getattr(symbol_info, "trade_tick_value", 0.0) or 1.0)
    vol_min = float(getattr(symbol_info, "volume_min", 0.01) or 0.01)
    vol_step = float(getattr(symbol_info, "volume_step", 0.01) or 0.01)
    vol_max = float(getattr(symbol_info, "volume_max", 100.0) or 100.0)

    loss_per_lot = (stop_distance / max(1e-9, tick_size)) * tick_value
    if loss_per_lot <= 0:
        return vol_min

    raw_lot = risk_amount / loss_per_lot

    # Round to volume_step
    steps = round(raw_lot / vol_step)
    calculated_lot = round(steps * vol_step, 4)
    return min(vol_max, max(vol_min, calculated_lot))


class PaperOrderTracker:
    """Tracks simulated live positions and manages TP1 partials, breakeven, and time-stop."""

    def __init__(self, storage_path: Path | str = DEFAULT_PAPER_PATH, notifier: ExecutionNotifier | None = None):
        self.path = Path(storage_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.notifier = notifier
        self._positions: list[dict[str, Any]] = self._load()

    def _load(self) -> list[dict[str, Any]]:
        if self.path.exists():
            try:
                return json.loads(self.path.read_text(encoding="utf-8"))
            except Exception:
                return []
        return []

    def _save(self) -> None:
        self.path.write_text(json.dumps(self._positions, indent=2, default=str) + "\n", encoding="utf-8")

    @property
    def open_positions(self) -> list[dict[str, Any]]:
        return [p for p in self._positions if p.get("status") == "OPEN"]

    def open_position(self, signal: OrderFlowSignal, lot: float = 1.0) -> dict[str, Any]:
        pos = {
            "id": f"paper_{len(self._positions)+1}_{int(datetime.now(timezone.utc).timestamp())}",
            "symbol": signal.symbol,
            "timeframe": signal.timeframe,
            "side": signal.side,
            "entry_price": signal.entry_price,
            "stop_loss": signal.stop_loss,
            "initial_stop": signal.stop_loss,
            "tp1": signal.tp1,
            "tp2": signal.tp2,
            "risk": signal.risk,
            "lot_size": lot,
            "qty_remaining": 1.0,
            "tp1_taken": False,
            "entry_time": signal.bar_time,
            "bars_held": 0,
            "status": "OPEN",
            "exit_time": None,
            "exit_price": None,
            "exit_reason": None,
            "realized_r": 0.0,
        }
        self._positions.append(pos)
        self._save()
        return pos

    def update(self, current_bar: Mapping[str, Any], current_time: datetime | pd.Timestamp) -> list[dict[str, Any]]:
        """Evaluate open positions against the latest closed bar."""
        updates = []
        lo = float(current_bar["low"])
        hi = float(current_bar["high"])
        close = float(current_bar["close"])

        for pos in self.open_positions:
            if pos["symbol"] != current_bar.get("symbol", pos["symbol"]):
                continue

            pos["bars_held"] += 1
            side = pos["side"]

            # 1. Friday weekend close (21:30 UTC) for non-crypto
            is_crypto = pos["symbol"].endswith("USDT")
            if not is_crypto and hasattr(current_time, "weekday") and current_time.weekday() == 4:
                if current_time.hour > 21 or (current_time.hour == 21 and current_time.minute >= 30):
                    self._close_position(pos, close, "WEEKEND_CLOSE", current_time)
                    updates.append(pos)
                    continue

            # 2. Stop Loss Hit
            stop_hit = (lo <= pos["stop_loss"]) if side == "LONG" else (hi >= pos["stop_loss"])
            if stop_hit:
                self._close_position(pos, pos["stop_loss"], "SL", current_time)
                updates.append(pos)
                continue

            # 3. TP1 Hit (50% partial take profit + move SL to breakeven)
            tp1_hit = (hi >= pos["tp1"]) if side == "LONG" else (lo <= pos["tp1"])
            if not pos["tp1_taken"] and tp1_hit:
                gross = 1.0  # Exactly 1R reached
                pos["realized_r"] += 0.5 * gross
                pos["qty_remaining"] = 0.5
                pos["tp1_taken"] = True
                pos["stop_loss"] = pos["entry_price"]  # Breakeven SL
                updates.append(pos)
                if self.notifier:
                    scoreboard = self.format_scoreboard_html()
                    msg = (
                        f"🎯 <b>[PAPER TP1 ALINDI & BREAKEVEN]</b>\n\n"
                        f"<b>Sembol:</b> <code>{pos['symbol']}</code> ({pos.get('timeframe', 'M5')}) | <b>Yön:</b> {side}\n"
                        f"<b>Kâr Alınan:</b> %50 (+0.50R)\n"
                        f"<b>Yeni SL:</b> <code>{pos['stop_loss']:.5f}</code> (Breakeven)\n\n"
                        f"{scoreboard}"
                    )
                    send_telegram_alert(msg, self.notifier.bot_token, self.notifier.chat_id)

            # 4. TP2 Hit (remaining 50% exit)
            tp2_hit = (hi >= pos["tp2"]) if side == "LONG" else (lo <= pos["tp2"])
            if pos["tp1_taken"] and tp2_hit:
                self._close_position(pos, pos["tp2"], "TP2", current_time)
                updates.append(pos)
                continue

            # 5. Time-Stop Hit (16 bars without TP1)
            if not pos["tp1_taken"] and pos["bars_held"] >= 16:
                self._close_position(pos, close, "TIME_STOP", current_time)
                updates.append(pos)
                continue

        self._save()
        return updates

    def get_monthly_stats(self, target_month: str | None = None) -> dict[str, Any]:
        """Calculate monthly performance statistics from paper positions."""
        if target_month is None:
            target_month = datetime.now(timezone.utc).strftime("%Y-%m")

        closed = [
            p for p in self._positions
            if p.get("status") == "CLOSED"
            and (
                str(p.get("exit_time", "")).startswith(target_month)
                or (not p.get("exit_time") and str(p.get("entry_time", "")).startswith(target_month))
            )
        ]

        total_closed = len(closed)
        net_r = sum(float(p.get("realized_r", 0.0)) for p in closed)

        tp_count = sum(1 for p in closed if p.get("exit_reason") == "TP2" or (float(p.get("realized_r", 0.0)) > 0.0 and p.get("tp1_taken")))
        sl_count = sum(1 for p in closed if p.get("exit_reason") == "SL" and not p.get("tp1_taken"))
        be_count = sum(1 for p in closed if p.get("tp1_taken") and p.get("exit_reason") == "SL")
        time_stop_count = sum(1 for p in closed if p.get("exit_reason") == "TIME_STOP")
        weekend_count = sum(1 for p in closed if p.get("exit_reason") == "WEEKEND_CLOSE")

        win_trades = sum(1 for p in closed if float(p.get("realized_r", 0.0)) > 0.0)
        win_rate = (win_trades / total_closed * 100.0) if total_closed > 0 else 0.0

        return {
            "month": target_month,
            "total_closed": total_closed,
            "net_r": net_r,
            "tp_count": tp_count,
            "sl_count": sl_count,
            "be_count": be_count,
            "time_stop_count": time_stop_count,
            "weekend_count": weekend_count,
            "win_rate": win_rate,
            "open_count": len(self.open_positions),
        }

    def format_scoreboard_html(self, target_month: str | None = None) -> str:
        """Format monthly performance scoreboard for Telegram and reports."""
        stats = self.get_monthly_stats(target_month)
        sign = "+" if stats["net_r"] >= 0 else ""
        return (
            f"📊 <b>[AYLIK PERFORMANS SAYACI — {stats['month']}]</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"💰 <b>Aylık Net R  :</b> <b>{sign}{stats['net_r']:.2f}R</b>\n"
            f"📈 <b>Kazanma Oranı :</b> %{stats['win_rate']:.1f} ({stats['tp_count']}/{stats['total_closed']} işlem)\n"
            f"🎯 <b>TP (Kâr Al)  :</b> {stats['tp_count']}\n"
            f"🛑 <b>SL (Zarar Durdur):</b> {stats['sl_count']}\n"
            f"🛡️ <b>BE (Başabaş) :</b> {stats['be_count']}\n"
            f"⏳ <b>Time-Stop    :</b> {stats['time_stop_count']}\n"
            f"📂 <b>Açık İşlem   :</b> {stats['open_count']}\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━"
        )

    def _close_position(self, pos: dict[str, Any], exit_price: float, reason: str, exit_time: Any) -> None:
        pos["status"] = "CLOSED"
        pos["exit_price"] = exit_price
        pos["exit_reason"] = reason
        pos["exit_time"] = str(exit_time)
        gross = ((exit_price - pos["entry_price"]) if pos["side"] == "LONG" else (pos["entry_price"] - exit_price)) / pos["risk"]
        pos["realized_r"] += pos["qty_remaining"] * gross
        pos["qty_remaining"] = 0.0

        if self.notifier:
            scoreboard = self.format_scoreboard_html()
            icon = "✅" if pos["realized_r"] > 0 else "❌"
            sign = "+" if pos["realized_r"] >= 0 else ""
            msg = (
                f"{icon} <b>[PAPER POZİSYON KAPANDI]</b>\n\n"
                f"<b>Sembol:</b> <code>{pos['symbol']}</code> ({pos.get('timeframe', 'M5')}) | <b>Yön:</b> {pos['side']}\n"
                f"<b>Gerekçe:</b> <code>{reason}</code>\n"
                f"<b>Çıkış Fiyatı:</b> <code>{exit_price:.5f}</code>\n"
                f"<b>İşlem Sonucu:</b> <b>{sign}{pos['realized_r']:+.2f}R</b>\n\n"
                f"{scoreboard}"
            )
            send_telegram_alert(msg, self.notifier.bot_token, self.notifier.chat_id)


class MT5OrderExecutor:
    """Manages live MT5 order placement, lot sizing, and open position lifecycle."""

    def __init__(self, mt5_module=None, magic: int = TRAWMA_MAGIC_NUMBER, risk_pct: float = 0.5):
        if mt5_module is None:
            import MetaTrader5 as mt5_module
        self.mt5 = mt5_module
        self.magic = magic
        self.risk_pct = risk_pct

    def place_order(self, signal: OrderFlowSignal) -> dict[str, Any] | None:
        """Place live MT5 order with calculated lot size and stops."""
        info = self.mt5.symbol_info(signal.symbol)
        if info is None:
            raise RuntimeError(f"Cannot retrieve MT5 symbol info for {signal.symbol}")
        acc = self.mt5.account_info()
        if acc is None:
            raise RuntimeError("Cannot retrieve MT5 account info")

        lot = calculate_mt5_lot_size(acc.balance, signal.entry_price, signal.stop_loss, info, risk_pct=self.risk_pct)

        order_type = self.mt5.ORDER_TYPE_BUY if signal.side == "LONG" else self.mt5.ORDER_TYPE_SELL
        price = info.ask if signal.side == "LONG" else info.bid

        request = {
            "action": self.mt5.TRADE_ACTION_DEAL,
            "symbol": signal.symbol,
            "volume": lot,
            "type": order_type,
            "price": price,
            "sl": float(signal.stop_loss),
            "tp": float(signal.tp2),
            "magic": self.magic,
            "comment": f"Trawma_{signal.divergence_type[:8]}",
            "type_time": self.mt5.ORDER_TIME_GTC,
            "type_filling": self.mt5.ORDER_FILLING_IOC,
        }

        result = self.mt5.order_send(request)
        if result is None or result.retcode != self.mt5.TRADE_RETCODE_DONE:
            err = result.comment if result else self.mt5.last_error()
            print(f"[MT5OrderExecutor Error] Order send failed: {err}", flush=True)
            return None

        print(f"[MT5OrderExecutor] Live Order OPENED: {signal.symbol} {signal.side} Lot: {lot} Ticket: {result.order}")
        return {"order": result.order, "lot": lot, "price": result.price}

    def close_open_positions_on_friday(self) -> int:
        """Enforces Weekend Gap Protection: closes all Trawma positions on Friday evening."""
        positions = self.mt5.positions_get()
        if not positions:
            return 0
        closed_count = 0
        for p in positions:
            if p.magic == self.magic:
                order_type = self.mt5.ORDER_TYPE_SELL if p.type == self.mt5.ORDER_TYPE_BUY else self.mt5.ORDER_TYPE_BUY
                price = self.mt5.symbol_info_tick(p.symbol).bid if p.type == self.mt5.ORDER_TYPE_BUY else self.mt5.symbol_info_tick(p.symbol).ask
                req = {
                    "action": self.mt5.TRADE_ACTION_DEAL,
                    "position": p.ticket,
                    "symbol": p.symbol,
                    "volume": p.volume,
                    "type": order_type,
                    "price": price,
                    "magic": self.magic,
                    "comment": "Trawma_WEEKEND_CLOSE",
                }
                res = self.mt5.order_send(req)
                if res and res.retcode == self.mt5.TRADE_RETCODE_DONE:
                    closed_count += 1
        return closed_count


class ExecutionBridge:
    """Unified bridge handling notify_only, paper, and mt5_live modes."""

    def __init__(
        self,
        mode: str = "notify_only",
        confirm_live: bool = False,
        risk_pct: float = 0.5,
        notifier: ExecutionNotifier | None = None,
        paper_path: Path | str = DEFAULT_PAPER_PATH,
    ):
        if mode not in ("notify_only", "paper", "mt5_live"):
            raise ValueError(f"Invalid mode: {mode}. Must be notify_only, paper, or mt5_live")

        if mode == "mt5_live" and not confirm_live:
            raise PermissionError("mt5_live mode requires explicit --confirm-live confirmation flag for account safety.")

        self.mode = mode
        self.risk_pct = risk_pct
        self.notifier = notifier or ExecutionNotifier()
        self.paper_tracker = PaperOrderTracker(storage_path=paper_path, notifier=self.notifier) if mode == "paper" else None
        self.mt5_executor = MT5OrderExecutor(risk_pct=risk_pct) if mode == "mt5_live" else None

    def execute_signal(self, signal: OrderFlowSignal) -> dict[str, Any]:
        """Dispatch signal according to selected mode."""
        # 1. Monthly Scoreboard
        scoreboard = None
        if self.paper_tracker:
            scoreboard = self.paper_tracker.format_scoreboard_html()
        elif Path(DEFAULT_PAPER_PATH).exists():
            try:
                scoreboard = PaperOrderTracker(storage_path=DEFAULT_PAPER_PATH).format_scoreboard_html()
            except Exception:
                scoreboard = None

        # 2. Always notify Telegram & Console with Scoreboard
        self.notifier.notify(signal, scoreboard=scoreboard)

        # 3. Mode dispatch
        if self.mode == "notify_only":
            return {"status": "NOTIFIED", "mode": "notify_only"}

        elif self.mode == "paper":
            pos = self.paper_tracker.open_position(signal, lot=1.0)
            return {"status": "PAPER_OPENED", "position": pos}

        elif self.mode == "mt5_live":
            result = self.mt5_executor.place_order(signal)
            return {"status": "LIVE_OPENED" if result else "LIVE_FAILED", "result": result}

        return {"status": "UNKNOWN"}
