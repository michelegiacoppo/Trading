@echo off
title XAU Quant Scalper - Installazione
cd /d "%~dp0"
echo ============================================================
echo   XAU Quant Scalper - installazione ambiente Python
echo ============================================================
if exist ".venv\Scripts\python.exe" goto install
echo Creo l'ambiente virtuale .venv ...
py -3.12 -m venv .venv >nul 2>nul
if not exist ".venv\Scripts\python.exe" py -3.11 -m venv .venv >nul 2>nul
if not exist ".venv\Scripts\python.exe" py -3 -m venv .venv >nul 2>nul
if not exist ".venv\Scripts\python.exe" python -m venv .venv
if not exist ".venv\Scripts\python.exe" (
  echo.
  echo ERRORE: Python non trovato.
  echo Installa Python 3.12 da https://www.python.org/downloads/
  echo e spunta "Add python.exe to PATH" durante l'installazione.
  pause
  exit /b 1
)
:install
".venv\Scripts\python.exe" --version
".venv\Scripts\python.exe" -m pip install --upgrade pip
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 (
  echo.
  echo ERRORE durante l'installazione dei pacchetti: copia il messaggio qui sopra.
  pause
  exit /b 1
)
echo.
echo Installazione completata. Ora apri MT5 (loggato) e lancia 2_DIAGNOSTICA.bat
pause
