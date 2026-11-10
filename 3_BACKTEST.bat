@echo off
title XAU Quant Scalper - Backtest
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (echo Lancia prima 1_INSTALLA.bat & pause & exit /b 1)
echo MT5 deve essere APERTO e LOGGATO.
echo.
set "SYMS="
set /p "SYMS=Simboli separati da spazio (Invio = tutti quelli abilitati in settings.yaml): "
set "SYMARG="
if not "%SYMS%"=="" set "SYMARG=--symbols %SYMS%"
set "PROF=base"
set /p "PROF=Profilo: base o attivo (Invio = base): "
set "DAYS=365"
set /p "DAYS=Quanti giorni di storico (Invio = 365): "
echo.
echo [1/2] Scarico/aggiorno lo storico da MT5 ...
".venv\Scripts\python.exe" main.py --mode download --days %DAYS% %SYMARG% --profile %PROF%
if errorlevel 1 (
  echo.
  echo Collegamento a MT5 non riuscito: leggi il suggerimento HINT qui sopra.
  echo Controlla che MT5 sia aperto e LOGGATO in un conto ^(anche demo^).
  pause
  exit /b 1
)
echo.
echo [2/2] Backtest ...
".venv\Scripts\python.exe" main.py --mode backtest --days %DAYS% %SYMARG% --profile %PROF%
echo.
echo Trade salvati in logs\backtest_trades.csv - equity in logs\equity_curve.csv
echo Copia TUTTO il testo di questa finestra e mandalo per l'analisi.
pause
