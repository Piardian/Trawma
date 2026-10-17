# Trawma

Trawma is an independent quantitative research project for order flow, liquidity-pool
detection, and CVD divergence research.

## Data semantics

The engine labels delta source explicitly:
- `exchange_volume_delta`: caller supplied `volume_delta`; the upstream definition must
  be verified to know whether it is true aggressor delta.
- `tick_rule_estimate`: signed tick volume estimated from tick-to-tick price changes.
- `bvc_synthetic_tick_volume` / `bvc_synthetic_volume`: OHLCV-based proxy, not actual
  bid/ask aggressor delta.

For credible tests, keep these sources separate. Do not compare BVC proxy values and exchange
trade delta as if they were the same measurement.

## Quick start

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
pytest -v
python run_pipeline.py --help
python backtest_engine.py --help
```

## Input data

CSV input should contain `timestamp,open,high,low,close`, plus one of:
`volume_delta`, `tick_volume`, or `volume`. Datetime values are interpreted as UTC if
timezone-naive. Optional OHLCV cache paths are not committed into this repository.

## Engines

- `amt`: reserved for a future independent AMT implementation.
- `orderflow`: liquidity pools from swings, session extremes, and effort/result candidates.
- `hybrid`: presently requires the AMT POI provider to be supplied by a future phase; until
  then it fails clearly instead of silently behaving like orderflow.

This repository starts from zero. There is no pre-existing AMT implementation or inherited
test suite in this repository yet.
