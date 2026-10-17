"""Trawma data-feature pipeline CLI."""
from .core import run_cli


def main() -> int:
    return run_cli("Trawma feature pipeline")


if __name__ == "__main__":
    raise SystemExit(main())
