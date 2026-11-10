# CLAUDE.md — XAU Quant Scalper

Context and rules for Claude Code working on this repository.

## What this is
Python fast-intraday / scalping framework for MetaTrader 5, focused on XAUUSD (plus US100, EURUSD,
AUDCAD). M1 trigger, M5 confirmation, M15 bias. Live trading only works on Windows (the
`MetaTrader5` package is Windows-only); backtest, tests and CI run on Linux.

## Commands
```bash
pip install -r requirements.txt
python -m pytest                                   # unit tests (live MT5 tests auto-skip)
ruff check . && ruff format --check .              # lint (CI runs both)
python main.py --mode backtest --synthetic         # smoke backtest, no data needed
python main.py --mode backtest --csv data/XAUUSD_M1.csv
python main.py --mode diag | dryrun | live --confirm-live   # Windows + MT5 terminal only
```

## Architecture rules (do not break)
1. **Lazy MT5 import.** Only `src/core/mt5_connector.py` imports `MetaTrader5`. Everything else gets
   the module via `MT5Connector.mt5`. Nothing at import time may require MT5.
2. **One strategy code path.** Strategies implement vectorised `compute_features()` (must output
   `signal` and `atr`) and `build_signal()`. Backtest and live both call these; never add a
   live-only or backtest-only signal branch.
3. **No look-ahead.** Bars are closed-only and stamped with *open* time. Multi-timeframe data must be
   joined with `indicators.align_htf` (as-of join on bar close time). `test_strategy_features_are_causal`
   and `test_align_htf_has_no_lookahead` must keep passing.
4. **Risk never rounds up.** `PositionSizer` floors to `volume_step` and refuses (0 lots) below
   `volume_min`. Spread and round-turn commission are part of the per-lot loss.
5. **Guardrails are pure + clock-injected.** Components in `src/risk/guardrails.py` take `now`
   explicitly so the backtester and tests reuse them unchanged. Don't call `datetime.now()` there.
6. **Trade management is pure.** `TradeManager.evaluate()` returns actions; `ExecutionEngine` (live),
   the paper book (dryrun) and `BacktestEngine` apply them. SL may only move in the trade's favour.
7. **Prices:** bars are BID. Longs enter on ASK and exit on BID; shorts enter on BID and are stopped
   on ASK (add spread). Times are broker server time (tz-naive).
8. **Safety:** `--mode live` requires `--confirm-live`. Never commit `.env`, credentials, CSV data or
   logs. Never weaken the kill-switch defaults without being asked.

## Conventions
- Python ≥ 3.10, full type hints, dataclasses for config/models, `logging` (no `print` in `src/`
  except Rich output in diagnostics/CLI).
- Config lives in `config/settings.yaml` → typed dataclasses in `src/config.py`. A new parameter needs
  both the YAML key and the dataclass field (with a sane default).
- New strategy: subclass `BaseStrategy` in `src/strategies/`, register it in `STRATEGIES`
  (`xau_velocity_scalp.py`), add a causality test.
- Line length 100, ruff rules in `pyproject.toml`. Run ruff format before committing.

## Key files
- `src/core/trader.py` — live/dryrun loop (orchestration only, no trading math)
- `src/core/execution.py` — order routing, filling mode, retries, TradeManager, PaperBook
- `src/risk/guardrails.py` — spread/gap/slippage/session/kill-switch
- `src/backtest/engine.py` — execution model documented in the module docstring
