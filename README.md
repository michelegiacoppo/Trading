# XAU Quant Scalper

[![CI](https://github.com/michelegiacoppo/Trading/actions/workflows/ci.yml/badge.svg)](https://github.com/michelegiacoppo/Trading/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Platform](https://img.shields.io/badge/MetaTrader-5-informational)

Framework Python di **scalping veloce / fast-intraday** a leva per **XAUUSD (Gold)** e CFD principali
(US100, EURUSD, AUDCAD…) su **MetaTrader 5**. Operazioni "Hit & Run" su M1 con filtro direzionale M15,
ingressi su micro-pullback e difesa sistematica dai problemi tipici dell'oro: micro-gap di liquidità,
slippage, stop-hunting e allargamenti improvvisi dello spread.

> ⚠️ **Disclaimer**: software a scopo di ricerca ed educativo. Il trading con leva su CFD comporta un
> rischio elevato di perdita del capitale. Usa sempre prima `dryrun` su conto demo.

---

## Strategia: XAU Velocity & Micro-Pullback Scalper

| Fase | Logica |
|---|---|
| **1. Bias M15** | EMA 9 > EMA 21 con prezzo oltre la EMA lenta, più l'inclinazione del **VWAP giornaliero** (`ema_vwap`) oppure massimi e minimi crescenti (`ema_structure`). Conferma opzionale con EMA M5 |
| **2. Trigger M1** | Candela di **Displacement** (corpo ≥ 1.5×ATR(14) **e** tick volume ≥ 1.5× la media), preceduta da un **Liquidity Sweep** (perforazione del max/min delle ultime N candele con chiusura di rientro) oppure in **rottura di uno Squeeze** (Bollinger dentro Keltner) |
| **3. Livelli istituzionali** | PDH/PDL, Asian High/Low (00-07 UTC), London High/Low e VWAP giornaliero ±1σ/±2σ. Lo sweep di un livello chiave ha **priorità assoluta**: `priority` = rischio pieno sui livelli chiave e ridotto sugli altri setup, `required` = si opera solo sui livelli chiave |
| **4. Smart Entry anti-gap** | `BUY_LIMIT`/`SELL_LIMIT` al ritracciamento del **35-50% della candela d'impulso** o sul filo del **micro Fair Value Gap**. Scadenza dopo `order_expiration_candles` (3 candele M1): se il prezzo scappa via l'ordine viene cancellato. Modalità Market con `max_deviation_points` (deviation MT5 e controllo anti-inseguimento) |
| **5. Stop strutturale** | Oltre l'estremo dello sweep + 0.3×ATR, mai sotto `sl_min_points`, spostato lontano dai numeri tondi (.00/.50). Sui short si aggiunge lo spread perché lo stop scatta sull'ASK |

### Gestione "Hit & Run"
1. **TP1 a +1.0R** (configurabile, es. 0.8R): chiusura immediata del **65%** e Stop Loss del restante 35%
   a **Break-Even + spread + commissioni**, quindi operazione a rischio zero.
2. **Micro-trailing del runner** sul minimo/massimo delle **ultime 2 candele M1** (oppure ATR), per
   catturare gli spike da 10-20 $ senza restituire il profitto.
3. **Stale Trade Killer**: dopo `max_trade_duration_minutes` (25') senza TP1, se il momentum si è spento
   la posizione viene chiusa a mercato vicino al pareggio.

### Risk & Guardrails
- **Lotti dinamici** calcolati da `symbol_info` (`trade_tick_value`, `trade_tick_size`, `volume_min/step`,
  `currency_profit`): la distanza entry→SL vale esattamente `risk_per_trade_pct` dell'equity, con spread
  e commissioni inclusi. L'arrotondamento è sempre per difetto e il trade viene rifiutato se servirebbe meno
  del lotto minimo.
- **Spread guard**: cap `max_spread_points`, controllo degli spike rispetto alla mediana mobile e del rapporto
  spread/ATR.
- **Gap/Jump detector**: un gap anomalo tra Close precedente e Open, oppure un salto tick-to-tick, blocca
  i nuovi ingressi per `gap_block_minutes`.
- **Sessioni in CET** (London 08:00-11:30, NY 14:00-18:30) con conversione automatica dall'ora server.
  **Rollover 22:45-01:15 bloccato**, così come weekend, apertura del lunedì, news ed eventuali slot di 30'
  esclusi.
- **Kill-switch giornaliero**: con perdita (chiusa + fluttuante) ≤ `-max_daily_loss_pct` (2.5%) chiude tutto,
  cancella i pendenti e passa in stato **`LOCKED`** fino al giorno successivo. Profit target opzionale e
  massimo 1-2 posizioni contemporanee.

### Connettore MT5
Rilevamento automatico del `filling_mode` (FOK / IOC / RETURN), **`order_check()` prima di ogni
`order_send()`**, log leggibile di ogni `retcode`, retry su requote e price-changed, `magic_number` dedicato
(i trade manuali non vengono toccati) e auto-reconnect.

---

## Multi-simbolo e profili

- **Simboli predefiniti** (abilitati in `config/settings.yaml`): XAUUSD, XAGUSD, EURUSD, GBPUSD, USDJPY,
  AUDUSD, USDCAD, EURJPY, GBPJPY, US100 (più USDCHF, NZDUSD, AUDCAD, US30, GER40 disattivati).
  Ognuno ha filtri spread, livelli tondi e limiti di stop loss propri (`sl_min_points` / `sl_max_points`).
- **Suffisso broker**: `general.symbol_suffix: ".m"` trasforma `EURUSD` in `EURUSD.m`. Si possono anche
  passare nomi come `GOLD` o `XAUUSD-ECN` con `--symbols`: ereditano i parametri del simbolo base.
- **Backtest di portafoglio**: ogni simbolo viene simulato e i risultati si sommano in un'unica curva di
  equity, con una tabella per simbolo che mostra dove la strategia funziona. Limite attuale: kill-switch e
  numero massimo di posizioni nel backtest valgono per singolo simbolo (nel live sono globali).
- **Profili** (`config/profiles/*.yaml`, sovrascrivono solo le chiavi indicate): `base` (selettivo) e
  `attivo` (trigger *pullback* di continuazione + breakout, finestra 08-21 CET, rischio 0,25% per trade).
  `--compare` li confronta sugli stessi dati.

## Laboratorio di strategie (`--mode lab`)

Un sistema che **genera, testa e scarta** strategie in automatico, progettato per difendersi
dall'errore più comune della ricerca quantitativa: trovare strategie "fortunate".

| Fase | Cosa fa |
|---|---|
| **Libreria** | 5 famiglie diverse: `trend_donchian`, `ema_cross`, `tsmom` (H1/H4), `orb` (rottura range apertura Londra/NY), `vwap_reversion` (M5/M15). Ognuna con griglia di parametri e gestione dell'uscita propria |
| **Ricerca** | Ogni famiglia × timeframe × parametri × simbolo, valutata **in pool su tutti i simboli** (una regola, non un adattamento per simbolo) |
| **Cassaforte** | Gli ultimi `holdout_months` (3) non vengono mai usati per scegliere: servono solo per la verifica finale, registrata nel journal |
| **Walk-forward** | Ad ogni passo sceglie le migliori configurazioni con i soli dati passati e le prova sui 3 mesi successivi: misura onestamente il *processo di selezione* |
| **Filtri anti-caso** | Trade minimi, expectancy positiva, **Deflated Sharpe Ratio** (più configurazioni provi, più alta l'asticella), robustezza dei parametri vicini, **stress test** costi (spread ×1.5, slippage ×2) |
| **Meta-labeling (ML)** | Per le sopravvissute, un modello `HistGradientBoosting` impara *quali segnali prendere* (contesto di mercato), validato con walk-forward purgato. Adottato solo se migliora l'expectancy fuori campione |
| **Portafoglio a conto unico** | Simulazione con equity condivisa, rischio composto, kill-switch e posizioni massime **globali** |
| **Promozione** | Solo chi supera tutto, cassaforte inclusa, diventa un profilo `config/profiles/lab_*.yaml` |
| **Live** | `--profiles lab` fa girare tutte le strategie promosse **insieme** (MultiTrader) con kill-switch condiviso, filtro ML e **monitor di salute** che spegne una strategia se i risultati reali divergono dal backtest. I trade chiusi finiscono in `logs/live_trades.csv` |
| **Imparare** | Rilanciare il laboratorio periodicamente (es. ogni mese): i dati nuovi rientrano nella ricerca e nei modelli |

```bash
python main.py --mode lab                          # tutte le famiglie, tutti i simboli abilitati
python main.py --mode lab --families orb tsmom --quick
python main.py --mode dryrun --profiles lab        # strategie promosse, in paper trading
```

## Dati storici & backtest anti-illusione

| Modulo | Cosa fa |
|---|---|
| **Downloader + cache Parquet** | `HistoricalStore.fetch_historical_data(symbol, tf, days_back)` usa `copy_rates_range` per M1/M5/M15/H1; `fetch_historical_ticks(symbol, start, end)` usa `copy_ticks_range(…, COPY_TICKS_ALL)` per i veri Bid/Ask. Tutto finisce in `data/historical/*.parquet` (ignorato da git) e i download successivi sono **incrementali**: solo le candele o i giorni di tick mancanti |
| **Cold-start warm-up** | All'avvio live scarica le ultime **1000 candele** di M1/M5/M15/H1 e costruisce la **mappa dei livelli istituzionali** prima di iniziare ad ascoltare il mercato |
| **Spread reale** | Il backtest usa la colonna `spread` delle barre MT5 oppure lo spread medio dei tick reali. Lo spread sintetico entra in gioco solo come ultima risorsa e il report lo segnala in rosso |
| **Slippage & commissioni** | `simulated_slippage_points` su ingressi Market, Stop Loss e uscite a mercato. Commissione round-turn per lotto. Gli stop in gap vengono eseguiti al peggiore tra SL e open |
| **Intra-bar conservativo** | Se SL e TP (o TP1) cadono nella stessa candela, **vince lo SL**, a meno che ci siano i tick per ricostruire il percorso reale del prezzo |
| **Report** | Win Rate, Profit Factor, Expectancy ($ e R), Max DD, Sharpe, Sortino, Recovery Factor, qualità d'esecuzione (fill rate dei limit, scaduti, blocchi) e **analisi oraria a slot di 30'** con suggerimento degli slot da escludere |
| **Auto-optimizer** | Grid search con selezione **in-sample** e validazione **out-of-sample** (`--optimize`) |

## Installazione

Il live richiede **Windows** con il terminale MetaTrader 5 installato e loggato (il pacchetto `MetaTrader5`
esiste solo per Windows). Backtest su CSV o cache, test e CI girano ovunque.

```bash
git clone https://github.com/michelegiacoppo/Trading.git xau-quant-scalper
```

```bash
cd xau-quant-scalper
```

```bash
python -m venv .venv
```

```bash
.venv\Scripts\activate
```

```bash
pip install -r requirements.txt
```

```bash
copy config\.env.example config\.env
```

Apri `config\.env` e inserisci `MT5_LOGIN`, `MT5_PASSWORD`, `MT5_SERVER` (ed eventualmente `MT5_PATH`).
Nel terminale MT5 abilita **Strumenti → Opzioni → Expert Advisor → "Consenti trading algoritmico"** e
il pulsante **AutoTrading**.

## Utilizzo

| Comando | Cosa fa |
|---|---|
| `python main.py --mode diag` | Diagnostica broker: conto, AutoTrading, tick_value, filling, storico, tick, offset server (nessun ordine) |
| `python main.py --mode download --days 180` | Riempie e aggiorna la cache Parquet (aggiungi `--ticks` per i tick reali) |
| `python main.py --mode backtest --days 60` | Scarica 60 giorni da MT5, esegue la simulazione e stampa il report |
| `python main.py --mode backtest --days 30 --ticks` | Come sopra, con tick reali per lo spread e il percorso intra-bar |
| `python main.py --mode backtest --days 60 --offline` | Usa solo la cache locale (anche su Linux/macOS) |
| `python main.py --mode backtest --days 365 --compare` | Confronta tutti i profili su tutti i simboli abilitati |
| `python main.py --mode backtest --days 365 --profile attivo` | Backtest con il profilo "attivo" (molti più trade) |
| `python main.py --mode backtest --days 365 --symbols XAUUSD EURUSD` | Solo i simboli indicati |
| `python main.py --mode backtest --days 90 --optimize` | Auto-optimizer IS/OOS (meglio su pochi simboli) |
| `python main.py --mode backtest --csv data/XAUUSD_M1.csv` | Backtest da export MT5, CSV o Parquet |
| `python main.py --mode dryrun` | Dati live, esecuzione su paper book, dashboard TUI |
| `python main.py --mode live --confirm-live` | Ordini **reali** |

Opzioni comuni: `--symbols XAUUSD US100`, `--no-ui`, `--config path/settings.yaml`.

## Struttura

```text
├── .github/workflows/ci.yml       # ruff + pytest (Python 3.10 / 3.12)
├── config/settings.yaml           # parametri commentati in italiano
├── config/.env.example            # template credenziali MT5
├── src/
│   ├── config.py                  # loader tipizzato YAML/.env
│   ├── core/
│   │   ├── mt5_connector.py       # connessione, reconnect, AutoTrading, filling, order_check/send, retcode
│   │   ├── data_feed.py           # OHLCV/tick live + HistoricalStore (download + cache Parquet incrementale)
│   │   ├── levels.py              # PDH/PDL, Asian/London range, VWAP ±σ (causali)
│   │   ├── clock.py               # ora server <-> UTC <-> CET
│   │   ├── execution.py           # Limit/Market, TP1 65% + BE, micro-trailing, stale killer, paper book
│   │   ├── trader.py              # warm-up + loop live/dryrun
│   │   ├── state.py               # snapshot thread-safe per la dashboard
│   │   └── diagnostics.py         # diagnostica broker
│   ├── risk/position_sizer.py     # lotti dinamici su equity / SL
│   ├── risk/guardrails.py         # spread, gap, slippage, sessioni CET, kill-switch LOCKED
│   ├── strategies/                # base astratta, indicatori, xau_velocity_scalp
│   ├── backtest/engine.py         # motore realistico (spread reale, slippage, intra-bar via tick)
│   ├── backtest/report.py         # report istituzionale rich + analisi oraria
│   ├── backtest/optimizer.py      # grid search IS/OOS
│   └── ui/dashboard.py            # Quant Desk TUI
├── tests/                         # unit test offline + diagnostica live
├── data/historical/               # cache Parquet (ignorata da git)
├── logs/                          # log runtime (ignorati da git)
└── main.py
```

## Dashboard "Quant Desk"

- **Header**: link MT5, simbolo, `LIVE` / `DRY-RUN`, sessione attiva (London/NY/Closed), ora CET.
- **Account & Risk**: balance, equity, margine libero, leva e margin level, P&L giornaliero (valuta e %),
  kill-switch `ACTIVE`/`LOCKED`.
- **Market Microstructure**: bid/ask, spread rispetto al massimo con semaforo, ATR M1/M5, volatilità,
  stato gap e livelli chiave più vicini.
- **Posizioni e ordini pendenti**: ticket, tipo, lotti, apertura, SL, TP, R, stato BE/TP1, profitto colorato.
- **Live Execution Log**: ultime 10 azioni con timestamp al millisecondo.

## Test

| Comando | Cosa fa |
|---|---|
| `python -m pytest` | Unit test offline (i test live MT5 vengono saltati) |
| `ruff check .` | Lint |
| `ruff format --check .` | Controllo formattazione |

Diagnostica con MT5 aperta: vedi `tests/test_mt5_diag.py` oppure la sezione dedicata in fondo a
`CLAUDE.md`.

## Licenza

MIT
