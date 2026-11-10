# CLAUDE.md — XAU Quant Scalper

Context and rules for Claude Code working on this repository.

## What this is
Python fast-intraday / scalping framework for MetaTrader 5, focused on XAUUSD (plus US100, EURUSD,
AUDCAD). M1 displacement trigger after a liquidity sweep / squeeze, M15 bias (EMA 9/21 + VWAP slope),
limit entries on micro-pullbacks, "Hit & Run" management (TP1 65% + BE, micro-trailing, stale killer).
Live trading only works on Windows (`MetaTrader5` is Windows-only); backtest from CSV/Parquet cache,
tests and CI run on Linux.

## Commands
```bash
pip install -r requirements.txt
python -m pytest                                   # offline unit tests (live MT5 tests auto-skip)
ruff check . && ruff format --check .              # lint (CI runs both)
python main.py --mode backtest --synthetic         # smoke backtest, no data needed
python main.py --mode backtest --days 60 [--ticks] [--optimize] [--offline]
python main.py --mode download --days 180 [--ticks]
python main.py --mode diag | dryrun | live --confirm-live   # Windows + MT5 terminal only
```

## Architecture rules (do not break)
1. **Lazy MT5 import.** Only `src/core/mt5_connector.py` imports `MetaTrader5`. Everything else gets
   the module via `MT5Connector.mt5`. Nothing at import time may require MT5.
2. **All trade requests go through `MT5Connector.send_request`** (order_check → order_send, retcode
   logging, retries). Filling mode comes from `MT5Connector.filling_mode`. Every request carries the
   bot `magic_number`; queries use `bot_positions` / `bot_orders` (never touch manual trades).
3. **One strategy code path.** Strategies implement vectorised `compute_features()` (must output
   `signal`, `atr`, `momentum`) and `build_signal()`. Backtest and live both call these.
4. **No look-ahead.** Bars are closed-only and stamped with *open* time. Higher-TF data and key levels
   are joined with `indicators.align_htf` (as-of join on bar close time). Keep
   `test_strategy_features_are_causal` and `test_align_htf_has_no_lookahead` green.
5. **Time zones.** Bars are broker server time. Use `ServerClock` (src/core/clock.py) to convert:
   sessions are in `sessions.timezone` (CET), key levels/day boundaries in UTC. Never compare server
   timestamps with CET strings directly.
6. **Risk never rounds up.** `PositionSizer` floors to `volume_step` and refuses below `volume_min`.
   Spread and round-turn commission are part of the per-lot loss.
7. **Guardrails & TradeManager are pure, clock-injected.** They take `now` explicitly and are reused
   unchanged by the backtester. SL may only move in the trade's favour. Partial exits are reported with
   `final=False` to the kill-switch.
8. **Backtest honesty.** Real spread first (bars `spread` column, then ticks); SL wins on intra-bar
   ambiguity unless ticks prove otherwise; slippage on market/stop fills. Don't add optimistic shortcuts.
9. **Prices:** bars are BID. Longs enter on ASK / exit on BID; shorts enter on BID / are stopped on ASK.
10. **Safety:** `--mode live` requires `--confirm-live`. Never commit `.env`, credentials, `data/`,
    `*.parquet`, CSVs or logs. Never weaken kill-switch defaults unless asked.

## Conventions
- Python ≥ 3.10, full type hints, dataclasses for config/models, `logging` (no `print` in `src/`
  except Rich output in diagnostics/report).
- Config: `config/settings.yaml` (Italian comments) → typed dataclasses in `src/config.py`. A new
  parameter needs both the YAML key and the dataclass field with a sane default.
- New strategy: subclass `BaseStrategy`, register in `STRATEGIES`, add a causality test.
- Line length 100; run `ruff format` before committing.

## Key files
- `src/core/trader.py` — warm-up + live/dryrun loop (orchestration only)
- `src/core/execution.py` — TradeManager (Hit & Run rules), ExecutionEngine, PaperBook
- `src/core/data_feed.py` — live feed + `HistoricalStore` (Parquet cache, incremental)
- `src/core/levels.py` — institutional levels
- `src/risk/guardrails.py` — spread/gap/slippage/session/kill-switch
- `src/backtest/engine.py` — execution model documented in the module docstring

## Running the live diagnostics (Windows, MT5 open)
```bat
python main.py --mode diag
python tests\test_mt5_diag.py XAUUSD
set RUN_MT5_DIAG=1 && python -m pytest tests\test_mt5_diag.py -v
```
