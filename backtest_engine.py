"""First-phase CLI entry for order-flow feature analysis (not yet a trade backtester)."""
from pragma.core import run_cli


def main() -> int:
    return run_cli("Trawma research/backtest entry point")


if __name__ == "__main__":
    raise SystemExit(main())
