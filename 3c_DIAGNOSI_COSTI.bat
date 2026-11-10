@echo off
title XAU Quant Scalper - Diagnosi costi (backtest a costi zero)
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (echo Lancia prima 1_INSTALLA.bat & pause & exit /b 1)
echo Backtest DIAGNOSTICO a costi zero (spread, slippage e commissioni = 0).
echo NON e' un risultato realizzabile: serve a capire se i segnali hanno valore prima dei costi.
echo Usa lo storico gia' scaricato (lancia prima 3_BACKTEST o 3b_CONFRONTA_PROFILI).
echo.
set "SYMS="
set /p "SYMS=Simboli separati da spazio (Invio = tutti quelli abilitati in settings.yaml): "
set "SYMARG="
if not "%SYMS%"=="" set "SYMARG=--symbols %SYMS%"
set "DAYS=365"
set /p "DAYS=Quanti giorni di storico (Invio = 365): "
".venv\Scripts\python.exe" main.py --mode backtest --days %DAYS% %SYMARG% --offline --compare --gross
echo.
echo Copia TUTTO il contenuto del file che si apre (Ctrl+A, Ctrl+C) e mandalo.
echo Report completo salvato in logs\ultimo_report.txt - lo apro in Blocco note.
if exist "logs\ultimo_report.txt" start "" notepad "logs\ultimo_report.txt"
pause
