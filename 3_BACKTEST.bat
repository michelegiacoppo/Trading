@echo off
title XAU Quant Scalper - Backtest
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (echo Lancia prima 1_INSTALLA.bat & pause & exit /b 1)
echo MT5 deve essere APERTO e LOGGATO.
echo.
set "SYM=XAUUSD"
set /p "SYM=Nome dell'oro nel tuo broker (Invio = XAUUSD): "
set "DAYS=365"
set /p "DAYS=Quanti giorni di storico (Invio = 365): "
echo.
echo [1/2] Scarico/aggiorno lo storico da MT5 ...
".venv\Scripts\python.exe" main.py --mode download --days %DAYS% --symbols %SYM%
echo.
echo [2/2] Backtest ...
".venv\Scripts\python.exe" main.py --mode backtest --days %DAYS% --symbols %SYM%
echo.
echo Trade salvati in logs\backtest_trades.csv - equity in logs\equity_curve.csv
pause
