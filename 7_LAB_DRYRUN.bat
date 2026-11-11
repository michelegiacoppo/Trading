@echo off
title XAU Quant Scalper - Strategie del laboratorio in DRY-RUN
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (echo Lancia prima 1_INSTALLA.bat & pause & exit /b 1)
if not exist "config\profiles\lab_*.yaml" (
  echo Nessuna strategia promossa dal laboratorio: lancia prima 6_LABORATORIO.bat
  pause
  exit /b 1
)
echo Fa girare INSIEME tutte le strategie promosse dal laboratorio (config\profiles\lab_*.yaml)
echo sullo stesso conto, con kill-switch condiviso e monitor di salute.
echo Esecuzione SIMULATA (paper): nessun ordine reale. Ctrl+C per fermare.
echo I trade chiusi vengono registrati in logs\live_trades.csv
".venv\Scripts\python.exe" main.py --mode dryrun --profiles lab
pause
