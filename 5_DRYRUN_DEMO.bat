@echo off
title XAU Quant Scalper - Dry-run (nessun ordine reale)
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
echo Dati live, esecuzione SIMULATA (paper). Ctrl+C per fermare.
".venv\Scripts\python.exe" main.py --mode dryrun %SYMARG% --profile %PROF%
pause
