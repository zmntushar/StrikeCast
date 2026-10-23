@echo off
REM ==========================================================================
REM  StrikeCast - start.bat
REM  Launches the StrikeCast dashboard and opens it in your web browser.
REM  Runs install.bat automatically the first time.
REM  Close this window (or press Ctrl+C) to stop the app.
REM ==========================================================================
setlocal
cd /d "%~dp0"
title StrikeCast

if not exist ".venv\Scripts\python.exe" (
    echo  First run: installing StrikeCast ...
    call "%~dp0install.bat" /nopause || (
        echo  [ERROR] Install failed. Run install.bat to see details.
        pause
        exit /b 1
    )
)

".venv\Scripts\python.exe" -c "import streamlit" >nul 2>nul || (
    echo  Packages are missing - running install.bat ...
    call "%~dp0install.bat" /nopause || (pause & exit /b 1)
)

echo.
echo  ===========================================
echo    StrikeCast is starting...
echo    Your browser will open automatically.
echo    If it does not, go to http://localhost:8501
echo    Close this window to stop the app.
echo  ===========================================
echo.
".venv\Scripts\python.exe" -m streamlit run dashboard\streamlit_app.py --server.headless false
if errorlevel 1 pause
