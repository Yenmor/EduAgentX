@echo off
setlocal
for %%I in ("%~dp0..") do set "PROJECT_ROOT=%%~fI"
cd /d "%PROJECT_ROOT%"
python -m uvicorn backend.app:app --host 127.0.0.1 --port 8002 > "%PROJECT_ROOT%\backend.webdemo8002.out.log" 2> "%PROJECT_ROOT%\backend.webdemo8002.err.log"
