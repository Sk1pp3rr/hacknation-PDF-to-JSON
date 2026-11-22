@echo off
chcp 65001 >nul
cd /d "%~dp0"

echo Uruchamianie aplikacji Odwolania2json...

if not exist ".venv\Scripts\python.exe" (
    echo [BLAD] Brak srodowiska .venv. Skontaktuj sie z administratorem IT.
    pause
    exit /b 1
)

if not exist "frontend\web\dist\index.html" (
    where npm >nul 2>nul
    if %ERRORLEVEL%==0 (
        echo Budowanie interfejsu webowego...
        pushd frontend\web
        call npm run build
        popd
    )
)

powershell -NoProfile -Command "try { $r = Invoke-WebRequest -UseBasicParsing -Uri 'http://127.0.0.1:8000/health' -TimeoutSec 2; if ($r.StatusCode -eq 200) { exit 0 } } catch {}; exit 1"
if %ERRORLEVEL%==0 (
    echo Aplikacja juz dziala. Otwieranie przegladarki...
    start "" "http://127.0.0.1:8000"
    exit /b 0
)

start "Odwolania2json - Serwer" /min cmd /c ".venv\Scripts\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8000"

echo Oczekiwanie na uruchomienie serwera...
for /L %%i in (1,1,20) do (
    powershell -NoProfile -Command "try { $r = Invoke-WebRequest -UseBasicParsing -Uri 'http://127.0.0.1:8000/health' -TimeoutSec 1; if ($r.StatusCode -eq 200) { exit 0 } } catch {}; exit 1"
    if not errorlevel 1 goto :ready
    timeout /t 1 /nobreak >nul
)

:ready
start "" "http://127.0.0.1:8000"
exit /b 0
