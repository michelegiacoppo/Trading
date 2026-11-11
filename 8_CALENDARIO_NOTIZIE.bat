@echo off
title XAU Quant Scalper - Calendario economico (notizie)
cd /d "%~dp0"
echo ============================================================
echo   Installa lo script che esporta il CALENDARIO ECONOMICO di MT5
echo   (dato reale, previsione, precedente) per il bot.
echo ============================================================
set "FOUND=0"
for /d %%D in ("%APPDATA%\MetaQuotes\Terminal\*") do (
  if exist "%%D\MQL5\Scripts" (
    copy /Y "mql5\XQS_ExportCalendar.mq5" "%%D\MQL5\Scripts\" >nul
    echo Copiato in: %%D\MQL5\Scripts
    set "FOUND=1"
  )
)
if "%FOUND%"=="0" (
  echo Cartella dati di MT5 non trovata. In MT5: File ^> Apri cartella dati,
  echo poi copia a mano mql5\XQS_ExportCalendar.mq5 in MQL5\Scripts
  pause
  exit /b 1
)
set "ME=C:\Program Files\MetaTrader 5\metaeditor64.exe"
if exist "%ME%" (
  for /d %%D in ("%APPDATA%\MetaQuotes\Terminal\*") do (
    if exist "%%D\MQL5\Scripts\XQS_ExportCalendar.mq5" "%ME%" /compile:"%%D\MQL5\Scripts\XQS_ExportCalendar.mq5" /log >nul 2>nul
  )
  echo Compilazione eseguita.
) else (
  echo MetaEditor non trovato nel percorso standard: apri MetaEditor ^(F4 in MT5^),
  echo apri Scripts\XQS_ExportCalendar.mq5 e premi F7 per compilarlo.
)
echo.
echo ORA IN MT5:
echo  1. Finestra Navigatore ^(Ctrl+N^) ^> Script ^> tasto destro ^> Aggiorna
echo  2. Trascina "XQS_ExportCalendar" su un grafico qualsiasi ^(es. XAUUSD^) e premi OK
echo  3. Lascialo attivo: aggiorna il file ogni 15 minuti ^(nella scheda Esperti vedi "events exported"^)
echo Il bot lo legge da solo: blocco notizie, informazioni per il ML e strategia news_drift.
pause
