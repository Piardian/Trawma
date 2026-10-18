"""Begonya Macro Bias Gate Reader for Trawma.

Integrates with Begonya's `macro_bias_gate.json` gateway file:
- Discovers candidate paths and selects the newest existing file.
- Handles UTC+3 naive timestamps written by Begonya.
- Respects gate precedence: EVENT_FREEZE / SYSTEMIC_STRESS > BASE_BIAS.
- Capital preservation mode shuts down all execution.
- Gating logic:
    * LONG_ONLY (+1): Only LONG signals permitted (Bullish Absorption + Positive Delta Flip).
    * SHORT_ONLY (-1): Only SHORT signals permitted (Bearish Absorption + Negative Delta Flip).
    * NEUTRAL_RANGE / EVENT_FREEZE / UNKNOWN (0): Execution strictly blocked.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
import json
import os
from pathlib import Path
from typing import Sequence

DEFAULT_CANDIDATE_PATHS: tuple[Path, ...] = (
    Path(r"C:\Users\piard\Desktop\Begonya00\macro_engine\gateways\macro_bias_gate.json"),
    Path(r"C:\Users\piard\Desktop\Begonya00\shared\macro_bias_gate.json"),
    Path("artifacts/macro_bias_gate.json"),
    Path("macro_bias_gate.json"),
)

SYMBOL_ALIASES: dict[str, str] = {
    "BTCUSDT": "BTC",
    "SOLUSDT": "SOL",
    "BTC": "BTC",
    "SOL": "SOL",
    "XAUUSD": "XAUUSD",
    "GOLD": "XAUUSD",
}

UTC_PLUS_3 = timezone(timedelta(hours=3))


@dataclass(frozen=True)
class MacroBiasDecision:
    symbol: str
    direction: int  # +1 (LONG), -1 (SHORT), 0 (BLOCKED/NEUTRAL)
    raw_gate: str
    allowed_side: str | None
    capital_preservation_mode: bool
    gate_file: str | None
    timestamp: datetime | None
    reason: str

    @property
    def is_executable(self) -> bool:
        return self.direction in (1, -1) and self.allowed_side is not None


def find_macro_bias_file(candidates: Sequence[Path] | None = None) -> Path | None:
    env_path = os.getenv("BEGONYA_MACRO_GATE_PATH")
    search_paths = [Path(env_path)] if env_path else []
    search_paths.extend(candidates or DEFAULT_CANDIDATE_PATHS)

    existing = [p for p in search_paths if p.exists() and p.is_file()]
    if not existing:
        return None
    # Pick the most recently modified file
    return max(existing, key=lambda p: p.stat().st_mtime)


def parse_begonya_timestamp(raw_ts: str | None) -> datetime | None:
    if not raw_ts:
        return None
    ts_str = str(raw_ts).strip()
    # Try ISO or standard datetime formats
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S%z", "%Y-%m-%dT%H:%M:%S%z"):
        try:
            dt = datetime.strptime(ts_str, fmt)
            if dt.tzinfo is None:
                # Begonya logs timestamps in local Turkish time (UTC+3)
                dt = dt.replace(tzinfo=UTC_PLUS_3)
            return dt.astimezone(timezone.utc)
        except ValueError:
            continue
    return None


def read_macro_bias(symbol: str, path: Path | str | None = None) -> MacroBiasDecision:
    norm_symbol = symbol.upper().replace("/", "").replace("-", "").replace("_", "")
    target_key = SYMBOL_ALIASES.get(norm_symbol, norm_symbol)

    resolved_path = Path(path) if path is not None else find_macro_bias_file()
    if resolved_path is None or not resolved_path.exists():
        return MacroBiasDecision(
            symbol=norm_symbol,
            direction=0,
            raw_gate="NO_GATE_FILE",
            allowed_side=None,
            capital_preservation_mode=False,
            gate_file=None,
            timestamp=None,
            reason="macro_bias_gate.json not found",
        )

    try:
        content = json.loads(resolved_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return MacroBiasDecision(
            symbol=norm_symbol,
            direction=0,
            raw_gate="PARSE_ERROR",
            allowed_side=None,
            capital_preservation_mode=False,
            gate_file=str(resolved_path),
            timestamp=None,
            reason=f"failed to parse macro gate json: {exc}",
        )

    ts = parse_begonya_timestamp(content.get("timestamp"))

    # Global capital preservation veto
    cap_pres = bool(content.get("capital_preservation_mode", False))
    if cap_pres:
        return MacroBiasDecision(
            symbol=norm_symbol,
            direction=0,
            raw_gate="CAPITAL_PRESERVATION",
            allowed_side=None,
            capital_preservation_mode=True,
            gate_file=str(resolved_path),
            timestamp=ts,
            reason="capital_preservation_mode active in macro engine",
        )

    # Check execution bias gates
    gates = content.get("execution_bias_gates") or content.get("deterministic_execution_bias_gates") or {}
    raw_gate = str(gates.get(target_key) or gates.get(norm_symbol) or "NEUTRAL_RANGE").upper().strip()

    # Precedence vetoes
    if any(freeze_word in raw_gate for freeze_word in ("FREEZE", "STRESS", "EVENT")):
        return MacroBiasDecision(
            symbol=norm_symbol,
            direction=0,
            raw_gate=raw_gate,
            allowed_side=None,
            capital_preservation_mode=False,
            gate_file=str(resolved_path),
            timestamp=ts,
            reason=f"macro gate indicates {raw_gate}",
        )

    if raw_gate == "LONG_ONLY":
        return MacroBiasDecision(
            symbol=norm_symbol,
            direction=1,
            raw_gate=raw_gate,
            allowed_side="LONG",
            capital_preservation_mode=False,
            gate_file=str(resolved_path),
            timestamp=ts,
            reason="LONG_ONLY macro alignment confirmed",
        )
    elif raw_gate == "SHORT_ONLY":
        return MacroBiasDecision(
            symbol=norm_symbol,
            direction=-1,
            raw_gate=raw_gate,
            allowed_side="SHORT",
            capital_preservation_mode=False,
            gate_file=str(resolved_path),
            timestamp=ts,
            reason="SHORT_ONLY macro alignment confirmed",
        )
    else:
        return MacroBiasDecision(
            symbol=norm_symbol,
            direction=0,
            raw_gate=raw_gate,
            allowed_side=None,
            capital_preservation_mode=False,
            gate_file=str(resolved_path),
            timestamp=ts,
            reason=f"macro bias is neutral/non-directional ({raw_gate})",
        )


def is_macro_aligned(symbol: str, side: str, path: Path | str | None = None) -> bool:
    """Check if the given trigger side aligns with Begonya's macro bias."""
    decision = read_macro_bias(symbol, path=path)
    if not decision.is_executable:
        return False
    return decision.allowed_side == side.upper()
