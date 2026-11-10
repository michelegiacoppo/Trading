@echo off
title XAU Quant Scalper - Diagnostica MT5
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (echo Lancia prima 1_INSTALLA.bat & pause & exit /b 1)
echo MT5 deve essere APERTO e LOGGATO. Nessun ordine viene inviato.
echo.
set "SYM=XAUUSD"
set /p "SYM=Nome dell'oro nel tuo broker (Invio = XAUUSD): "
".venv\Scripts\python.exe" main.py --mode diag --symbols %SYM%
echo.
pause
