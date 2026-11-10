@echo off
title XAU Quant Scalper - Confronto profili
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (echo Lancia prima 1_INSTALLA.bat & pause & exit /b 1)
echo MT5 deve essere APERTO e LOGGATO.
echo.
set "SYMS="
set /p "SYMS=Simboli separati da spazio (Invio = tutti quelli abilitati in settings.yaml): "
set "SYMARG="
if not "%SYMS%"=="" set "SYMARG=--symbols %SYMS%"
set "DAYS=365"
set /p "DAYS=Quanti giorni di storico (Invio = 365): "
echo.
echo [1/2] Scarico/aggiorno lo storico da MT5 ...
".venv\Scripts\python.exe" main.py --mode download --days %DAYS% %SYMARG%
if errorlevel 1 (echo Collegamento a MT5 non riuscito. & pause & exit /b 1)
echo.
echo [2/2] Confronto di tutti i profili sugli stessi dati ...
".venv\Scripts\python.exe" main.py --mode backtest --days %DAYS% %SYMARG% --compare
echo.
echo Copia TUTTO il testo di questa finestra e mandalo per l'analisi.
pause
