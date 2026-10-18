@echo off
cd /d "C:\Users\piard\Desktop\TrawmaProject"
title Trawma - Order Flow (Absorption-Only) Sentinel
color 0b
echo ======================================================================
echo  TRAWMA v1.0 ORDER FLOW TICARET SISTEMI BASLATILIYOR...
echo  - Mod: Paper Trading (Sanal Islem Defteri)
echo  - Izleme: SOLUSDT (M15), XAUUSD (M5)
echo  - Cikis: 16-Bar Time-Stop + Cuma Hafta Sonu Gap Korumasi
echo ======================================================================

set PYTHON_EXE=C:\Users\piard\AppData\Local\Programs\Python\Python312\python.exe

"%PYTHON_EXE%" -u run_pipeline.py --scan-begonya --scan-crypto --refresh-macro --watch --interval-sec 15 --execute-mode paper

pause
