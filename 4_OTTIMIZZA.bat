@echo off
title XAU Quant Scalper - Auto-optimizer
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (echo Lancia prima 1_INSTALLA.bat & pause & exit /b 1)
echo MT5 deve essere APERTO e LOGGATO.
echo.
set "SYMS=XAUUSD"
set /p "SYMS=Simboli (Invio = XAUUSD; ogni simbolo aggiunge ~3 minuti): "
set "PROF=base"
set /p "PROF=Profilo: base o attivo (Invio = base): "
set "DAYS=365"
set /p "DAYS=Quanti giorni di storico (Invio = 365): "
echo.
echo Ottimizzazione: sceglie i parametri sul 70%% dei dati e li verifica sul 30%% restante (OOS).
".venv\Scripts\python.exe" main.py --mode backtest --days %DAYS% --symbols %SYMS% --profile %PROF% --optimize
echo.
echo Guarda SOLO le colonne OOS: devono restare positive.
echo Report completo salvato in logs\ultimo_report.txt - lo apro in Blocco note.
if exist "logs\ultimo_report.txt" start "" notepad "logs\ultimo_report.txt"
pause
