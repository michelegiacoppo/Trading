@echo off
title XAU Quant Scalper - Auto-optimizer
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (echo Lancia prima 1_INSTALLA.bat & pause & exit /b 1)
set "SYM=XAUUSD"
set /p "SYM=Nome dell'oro nel tuo broker (Invio = XAUUSD): "
set "DAYS=365"
set /p "DAYS=Quanti giorni di storico (Invio = 365): "
echo.
echo Ottimizzazione: sceglie i parametri sul 70%% dei dati e li verifica sul 30%% restante (OOS).
".venv\Scripts\python.exe" main.py --mode backtest --days %DAYS% --symbols %SYM% --optimize
echo.
echo Guarda SOLO le colonne OOS: devono restare positive.
pause
