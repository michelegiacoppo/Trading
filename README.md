# XAU Quant Scalper

[![CI](https://github.com/michelegiacoppo/Trading/actions/workflows/ci.yml/badge.svg)](https://github.com/michelegiacoppo/Trading/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Platform](https://img.shields.io/badge/MetaTrader-5-informational)

Framework Python di **scalping veloce / fast-intraday** a leva per **XAUUSD (Gold)** e CFD principali
(US100, EURUSD, AUDCAD…) su **MetaTrader 5**, con execution "Hit & Run" su M1/M5, filtro direzionale M15
e una difesa sistematica dai problemi tipici dell'oro: micro-gap di liquidità, slippage, stop-hunting
e allargamenti improvvisi dello spread.

> ⚠️ **Disclaimer** — Software a scopo di ricerca/educativo. Il trading con leva su CFD comporta un
> rischio elevato di perdita del capitale. Testa sempre in `dryrun` e su conto demo prima del live.

---

## Caratteristiche

| Area | Cosa fa |
|---|---|
| **Connessione** | `initialize`/login da `.env`, auto-reconnect con back-off esponenziale, check del pulsante AutoTrading, dei permessi account/EA e dell'API Python |
| **Data feed** | OHLCV M1/M5/M15 **solo barre chiuse** (niente decisioni su candele in formazione), tick snapshot con spread in punti, rilevamento nuova barra a costo minimo |
| **Strategia** | `XauVelocityScalp`: bias M15 (EMA50 + slope) → conferma M5 (EMA9/21) → trigger M1 su *burst di velocità* (≥ k·ATR) con corpo forte, close sull'estremo, spike di tick-volume, RSI non esausto |
| **Anti-gap / anti-hunt** | Scarta barre-trigger > 3·ATR (news spike) e barre con gap in apertura; SL oltre lo swing del burst **e** spostato oltre i numeri tondi (`.00`/`.50`) dove si accumula la liquidità degli stop; spread aggiunto agli SL dei short (stoppati sull'ASK) |
| **Execution** | Market o **Limit su pullback** con scadenza (fill migliore, meno slippage), filling mode auto (FOK/IOC/RETURN), `order_check` pre-flight, retry su requote/price-changed, rispetto di `stops_level`/`freeze_level` |
| **Trade management** | Chiusura parziale a +1R, Break-Even (+buffer), trailing ATR, time-stop per gli scalp che non partono |
| **Risk** | Lotti dinamici su equity × rischio% / distanza SL (spread + commissioni inclusi), arrotondamento **sempre per difetto**, rifiuto se < lotto minimo, cap su lotti e utilizzo margine |
| **Guardrails** | Spread filter dinamico (cap assoluto + mediana mobile + rapporto spread/ATR), gap detector su barre e tick, monitor slippage con pausa automatica, watchdog feed "stale", sessioni/rollover/weekend/news blackout |
| **Kill-switch giornaliero** | Perdita giornaliera %, drawdown trailing intraday dal picco, N perdite consecutive, max trade/giorno, cooldown post-perdita, profit-lock opzionale |
| **Backtest** | Motore event-driven veloce (≈40k barre M1/s): spread per barra, commissioni, slippage, SL-first su ambiguità intrabar, gap-through degli stop, stessi guardrail e stessa logica di gestione del live |
| **Dashboard** | TUI live con `rich`: scanner mercato, conto & rischio, posizioni con R in tempo reale, log eventi |

## Architettura

```text
                ┌────────────────────┐
  settings.yaml │   src/config.py    │ .env (credenziali)
                └─────────┬──────────┘
                          │
 ┌───────────────┐   ┌────▼──────────────┐   ┌─────────────────────┐
 │ MT5Connector  │──▶│  LiveTrader loop  │──▶│ Dashboard (rich TUI)│
 │ (reconnect,   │   │  src/core/trader  │   │  thread separato    │
 │  AutoTrading) │   └─┬───────┬───────┬─┘   └─────────────────────┘
 └──────┬────────┘     │       │       │
        │        ┌─────▼──┐ ┌──▼─────┐ ┌▼────────────────┐
        └───────▶│DataFeed│ │Strategy│ │ ExecutionEngine │
                 └────────┘ └──┬─────┘ │ + TradeManager  │
                               │       └────────▲────────┘
                     ┌─────────▼─────────┐      │
                     │ Guardrails        │──────┘
                     │ + PositionSizer   │
                     └───────────────────┘
       BacktestEngine riusa: Strategy · Guardrails · PositionSizer · TradeManager
```

**Principio chiave:** la strategia espone un solo metodo vettoriale `compute_features()`. Il backtester lo
chiama una volta su tutta la storia, il live sull'ultima finestra leggendo l'ultima riga → **stesso codice,
nessuna divergenza live/backtest**. L'allineamento multi-timeframe usa i tempi di *chiusura* delle barre
(`align_htf`), quindi è garantita l'assenza di look-ahead (coperta da test).

## Struttura

```text
├── .github/workflows/ci.yml     # ruff + pytest
├── config/
│   ├── settings.yaml            # strategia, rischio, sessioni, filtri spread/gap
│   └── .env.example             # template credenziali MT5
├── src/
│   ├── config.py                # loader tipizzato YAML/.env
│   ├── core/
│   │   ├── mt5_connector.py     # connessione, auto-reconnect, check AutoTrading
│   │   ├── data_feed.py         # OHLCV M1/M5/M15 + tick
│   │   ├── execution.py         # ordini Market/Limit, partial, BE, trailing, paper book
│   │   ├── trader.py            # loop live/dryrun
│   │   ├── state.py             # snapshot thread-safe per la dashboard
│   │   └── diagnostics.py       # diagnostica broker
│   ├── risk/
│   │   ├── position_sizer.py    # lotti dinamici su equity/ATR SL
│   │   └── guardrails.py        # spread, gap, slippage, sessioni, kill-switch
│   ├── strategies/
│   │   ├── base_strategy.py     # contratto astratto + Signal
│   │   ├── indicators.py        # EMA/ATR/RSI vettoriali, align multi-TF
│   │   └── xau_velocity_scalp.py
│   ├── backtest/engine.py
│   └── ui/dashboard.py
├── tests/
│   ├── test_risk.py             # unit test sizing/guardrail/management/no-lookahead
│   └── test_mt5_diag.py         # diagnostica live (skippata in CI)
├── logs/                        # log runtime (ignorati da git)
├── main.py                      # CLI
├── CLAUDE.md
└── requirements.txt
```

## Installazione

Il live richiede **Windows** con terminale MetaTrader 5 installato e loggato (il pacchetto `MetaTrader5`
esiste solo per Windows). Backtest, test e CI girano ovunque.

```bash
git clone https://github.com/michelegiacoppo/Trading.git xau-quant-scalper
```

```bash
cd xau-quant-scalper && python -m venv .venv
```

```bash
.venv\Scripts\activate        # Windows  (Linux/macOS: source .venv/bin/activate)
```

```bash
pip install -r requirements.txt
```

```bash
copy config\.env.example config\.env   # poi inserisci login/password/server
```

Nel terminale MT5: **Strumenti → Opzioni → Expert Advisor → "Consenti trading algoritmico"** e pulsante
**AutoTrading** attivo.

## Utilizzo

```bash
python main.py --mode diag                       # verifica broker, tick_value, filling mode, AutoTrading
```

```bash
python main.py --mode dryrun                     # dati live, esecuzione su paper book, dashboard TUI
```

```bash
python main.py --mode live --confirm-live        # ordini REALI (richiede il flag esplicito)
```

```bash
python main.py --mode backtest --csv data/XAUUSD_M1.csv
```

```bash
python main.py --mode backtest --synthetic --bars 60000   # smoke test senza dati
```

Opzioni utili: `--symbols XAUUSD US100`, `--no-ui` (log a console), `--config path/settings.yaml`.

### Dati per il backtest

Esporta da MT5 (*Visualizza → Simboli → Barre*, timeframe M1) oppure fornisci un CSV con colonne
`time,open,high,low,close[,tick_volume,spread]`. I prezzi sono BID; la colonna `spread` (in punti) viene
usata barra per barra, altrimenti si usa `backtest.default_spread_points` + jitter casuale. Output in
`logs/equity_curve.csv` e `logs/backtest_trades.csv`.

## Configurazione (estratto)

```yaml
risk:
  risk_per_trade_pct: 0.50     # % equity alla distanza SL
  daily_loss_limit_pct: 3.0    # kill-switch
  max_consecutive_losses: 4
symbols:
  XAUUSD:
    max_spread_points: 35      # cap assoluto
    spread_median_mult: 1.8    # spike vs mediana mobile
    spread_atr_ratio_max: 0.15 # spread <= 15% ATR(M1)
    stop_hunt_round_step: 0.50 # evita SL vicino a .00/.50
strategy:
  velocity_atr_mult: 1.2       # burst minimo in multipli di ATR
  entry_mode: limit            # market | limit (pullback)
  tp_r_multiple: 1.6
```

Tutti gli orari di sessione sono in **ora server del broker** (come i timestamp delle barre MT5).

## Test & qualità

```bash
python -m pytest            # unit test (i test live MT5 sono skippati)
```

```bash
ruff check . && ruff format --check .
```

Diagnostica live sul PC di trading:

```bash
set RUN_MT5_DIAG=1 && python -m pytest tests/test_mt5_diag.py
```

## Note sul modello di esecuzione del backtest

- Segnale sulla chiusura della barra *i* → ordine dalla barra *i+1*.
- Market: fill all'open successivo + spread + slippage avverso.
- Limit: fill solo se il prezzo **attraversa** il livello entro la scadenza.
- Se SL e TP sono toccati nella stessa barra vince **lo SL**; gli stop in gap vengono eseguiti al peggiore
  tra SL e open.
- Il risultato su dati sintetici è privo di significato statistico: serve solo come smoke test.

## Licenza

MIT
