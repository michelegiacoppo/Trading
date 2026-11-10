@echo off
title XAU Quant Scalper - Installazione
cd /d "%~dp0"
echo ============================================================
echo   XAU Quant Scalper - installazione ambiente Python
echo ============================================================
if exist ".venv\Scripts\python.exe" goto check
echo Creo l'ambiente virtuale .venv ...
py -3.12-64 -m venv .venv >nul 2>nul
if not exist ".venv\Scripts\python.exe" py -3.12 -m venv .venv >nul 2>nul
if not exist ".venv\Scripts\python.exe" py -3.13-64 -m venv .venv >nul 2>nul
if not exist ".venv\Scripts\python.exe" py -3.11-64 -m venv .venv >nul 2>nul
if not exist ".venv\Scripts\python.exe" py -3-64 -m venv .venv >nul 2>nul
if not exist ".venv\Scripts\python.exe" py -3 -m venv .venv >nul 2>nul
if not exist ".venv\Scripts\python.exe" python -m venv .venv
if not exist ".venv\Scripts\python.exe" goto nopython

:check
rem MetaTrader5 is published only for 64-bit Intel/AMD Windows Python (win-amd64)
".venv\Scripts\python.exe" -c "import sys,sysconfig;p=sysconfig.get_platform();print('Python',sys.version.split()[0],'-',p);sys.exit(0 if p=='win-amd64' else 3)"
if errorlevel 3 goto wrongpython

".venv\Scripts\python.exe" -m pip install --upgrade pip
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto piperror
echo.
echo Installazione completata. Ora apri MT5 (loggato) e lancia 2_DIAGNOSTICA.bat
pause
exit /b 0

:wrongpython
echo.
echo ERRORE: questo Python NON e' "64-bit Intel/AMD" (win-amd64).
echo Il pacchetto MetaTrader5 esiste solo per quella versione.
echo.
echo SOLUZIONE:
echo  1. Vai su https://www.python.org/downloads/windows/
echo  2. Scarica Python 3.12.x "Windows installer (64-bit)"
echo     ATTENZIONE: NON "32-bit" e NON "ARM64".
echo     Anche se il PC ha un processore ARM/Snapdragon usa la 64-bit:
echo     funziona in emulazione, esattamente come MetaTrader 5.
echo  3. Installa spuntando "Add python.exe to PATH"
echo  4. Rilancia questo file.
echo.
echo Rimuovo l'ambiente creato con il Python sbagliato...
rmdir /s /q .venv
pause
exit /b 1

:nopython
echo.
echo ERRORE: Python non trovato.
echo Installa Python 3.12 "Windows installer (64-bit)" da https://www.python.org/downloads/windows/
echo e spunta "Add python.exe to PATH" durante l'installazione.
pause
exit /b 1

:piperror
echo.
echo ERRORE durante l'installazione dei pacchetti: copia tutto il messaggio qui sopra.
pause
exit /b 1
