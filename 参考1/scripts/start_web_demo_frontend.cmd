@echo off
setlocal
for %%I in ("%~dp0..") do set "PROJECT_ROOT=%%~fI"
cd /d "%PROJECT_ROOT%\frontend"
set NEXT_PUBLIC_API_BASE_URL=http://127.0.0.1:8002
npm.cmd run dev -- --hostname 127.0.0.1 --port 3001 > "%PROJECT_ROOT%\frontend\frontend.webdemo3001.out.log" 2> "%PROJECT_ROOT%\frontend\frontend.webdemo3001.err.log"
