//+------------------------------------------------------------------+
//| XQS_ExportCalendar.mq5                                           |
//| Exports the MT5 economic calendar (actual / forecast / previous) |
//| to <Common\Files>\xqs_calendar.csv for the XAU Quant Scalper.    |
//| Times are trade-server time (same as the chart bars).            |
//|                                                                  |
//| Usage: drag the script onto any chart. With InpKeepUpdating=true |
//| it stays active and refreshes the file every InpRefreshMinutes,  |
//| so the Python bot always knows the upcoming releases.            |
//+------------------------------------------------------------------+
#property copyright "XAU Quant Scalper"
#property version   "1.00"
#property script_show_inputs

input int    InpYearsBack       = 6;      // years of history to export
input int    InpDaysAhead       = 14;     // upcoming days to export
input int    InpMinImportance   = 2;      // 1=low 2=moderate 3=high
input bool   InpKeepUpdating    = true;   // keep running and refresh the file
input int    InpRefreshMinutes  = 15;     // refresh period
input string InpFileName        = "xqs_calendar.csv";

string Currencies[] = {"USD","EUR","GBP","JPY","AUD","CAD","CHF","NZD","CNY"};

string Num(const MqlCalendarValue &v, const int which)
  {
   if(which==0 && v.HasActualValue())   return DoubleToString(v.GetActualValue(), 6);
   if(which==1 && v.HasForecastValue()) return DoubleToString(v.GetForecastValue(), 6);
   if(which==2 && v.HasPreviousValue()) return DoubleToString(v.GetPreviousValue(), 6);
   return "";
  }

string Clean(string s)
  {
   StringReplace(s, "\t", " ");
   StringReplace(s, "\r", " ");
   StringReplace(s, "\n", " ");
   return s;
  }

bool ExportOnce()
  {
   datetime now  = TimeTradeServer();
   datetime from = now - (datetime)InpYearsBack * 365 * 86400;
   datetime to   = now + (datetime)InpDaysAhead * 86400;
   string tmp = InpFileName + ".tmp";
   int h = FileOpen(tmp, FILE_WRITE | FILE_TXT | FILE_COMMON | FILE_ANSI);
   if(h == INVALID_HANDLE)
     {
      Print("XQS calendar: FileOpen failed, error ", GetLastError());
      return false;
     }
   FileWriteString(h, "time\tcurrency\tevent\timportance\tactual\tforecast\tprevious\timpact\r\n");
   int rows = 0;
   for(int c = 0; c < ArraySize(Currencies); c++)
     {
      MqlCalendarValue values[];
      if(CalendarValueHistory(values, from, to, NULL, Currencies[c]) <= 0)
         continue;
      for(int i = 0; i < ArraySize(values); i++)
        {
         MqlCalendarEvent ev;
         if(!CalendarEventById(values[i].event_id, ev))
            continue;
         if((int)ev.importance < InpMinImportance)
            continue;
         string line = TimeToString(values[i].time, TIME_DATE | TIME_MINUTES) + "\t" + Currencies[c] + "\t" +
                       Clean(ev.name) + "\t" + IntegerToString((int)ev.importance) + "\t" +
                       Num(values[i], 0) + "\t" + Num(values[i], 1) + "\t" + Num(values[i], 2) + "\t" +
                       IntegerToString((int)values[i].impact_type) + "\r\n";
         FileWriteString(h, line);
         rows++;
        }
     }
   FileClose(h);
   // atomic replace so the bot never reads a half-written file
   FileDelete(InpFileName, FILE_COMMON);
   if(!FileMove(tmp, FILE_COMMON, InpFileName, FILE_COMMON | FILE_REWRITE))
      Print("XQS calendar: FileMove failed, error ", GetLastError());
   PrintFormat("XQS calendar: %d events exported to Common\\Files\\%s", rows, InpFileName);
   return rows > 0;
  }

void OnStart()
  {
   ExportOnce();
   while(InpKeepUpdating && !IsStopped())
     {
      for(int s = 0; s < InpRefreshMinutes * 60 && !IsStopped(); s++)
         Sleep(1000);
      if(!IsStopped())
         ExportOnce();
     }
  }
//+------------------------------------------------------------------+
