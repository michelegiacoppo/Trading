@echo off
title XAU Quant Scalper - Dry-run (nessun ordine reale)
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (echo Lancia prima 1_INSTALLA.bat & pause & exit /b 1)
set "SYM=XAUUSD"
set /p "SYM=Nome dell'oro nel tuo broker (Invio = XAUUSD): "
echo Dati live, esecuzione SIMULATA (paper). Ctrl+C per fermare.
".venv\Scripts\python.exe" main.py --mode dryrun --symbols %SYM%
pause
