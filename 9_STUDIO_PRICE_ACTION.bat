@echo off
title XAU Quant Scalper - Studio di Price Action
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (echo Lancia prima 1_INSTALLA.bat & pause & exit /b 1)
echo ============================================================
echo   STUDIO DI PRICE ACTION: per ogni pattern a candele giapponesi misura
echo   quanto ha reso davvero sui tuoi dati, su supporti/resistenze o no,
echo   con o contro il trend, con o senza conferma (spread incluso).
echo ============================================================
echo MT5 deve essere APERTO e LOGGATO (usa lo storico gia' scaricato dal laboratorio).
echo.
set "SYMS="
set /p "SYMS=Simboli separati da spazio (Invio = tutti quelli abilitati in settings.yaml): "
set "SYMARG="
if not "%SYMS%"=="" set "SYMARG=--symbols %SYMS%"
".venv\Scripts\python.exe" main.py --mode study %SYMARG%
echo.
echo Report completo in logs\ultimo_studio_price_action.txt - lo apro in Blocco note.
echo Copia TUTTO il contenuto del file (Ctrl+A, Ctrl+C) e mandalo.
if exist "logs\ultimo_studio_price_action.txt" start "" notepad "logs\ultimo_studio_price_action.txt"
pause
