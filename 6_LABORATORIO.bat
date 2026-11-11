@echo off
title XAU Quant Scalper - Laboratorio di strategie
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (echo Lancia prima 1_INSTALLA.bat & pause & exit /b 1)
echo ============================================================
echo   LABORATORIO: cerca strategie su tutte le famiglie, simboli e timeframe
echo   con walk-forward, filtri anti-caso, stress test, machine learning
echo   e verifica finale sulla "cassaforte" (ultimi 3 mesi mai usati).
echo ============================================================
echo MT5 deve essere APERTO e LOGGATO. Il primo avvio scarica anni di storico
echo (M5, M15, H1, H4): puo' richiedere 10-30 minuti.
echo.
".venv\Scripts\python.exe" -m pip install -q -r requirements.txt
set "SYMS="
set /p "SYMS=Simboli separati da spazio (Invio = tutti quelli abilitati in settings.yaml): "
set "SYMARG="
if not "%SYMS%"=="" set "SYMARG=--symbols %SYMS%"
".venv\Scripts\python.exe" main.py --mode lab %SYMARG%
echo.
echo Report completo salvato in logs\ultimo_laboratorio.txt - lo apro in Blocco note.
echo Copia TUTTO il contenuto del file (Ctrl+A, Ctrl+C) e mandalo.
if exist "logs\ultimo_laboratorio.txt" start "" notepad "logs\ultimo_laboratorio.txt"
pause
