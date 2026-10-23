@echo off
REM ==========================================================================
REM  StrikeCast - install.bat
REM  Creates a local Python virtual environment in .venv (inside this folder)
REM  and installs every package StrikeCast needs. Nothing is installed
REM  system-wide. Safe to run again: it just updates the packages.
REM  Requires Python 3.12 or newer: https://www.python.org/downloads/
REM ==========================================================================
setlocal
cd /d "%~dp0"
title StrikeCast - Install

echo.
echo  ===========================================
echo    StrikeCast installer
echo  ===========================================
echo.

REM ---- find Python 3.12+ ---------------------------------------------------
set "PY="
py -3.12 -c "import sys" >nul 2>nul && set "PY=py -3.12"
if not defined PY (
    python -c "import sys; sys.exit(0 if sys.version_info >= (3, 12) else 1)" >nul 2>nul && set "PY=python"
)
if not defined PY (
    echo  [ERROR] Python 3.12 or newer was not found.
    echo          Install it from https://www.python.org/downloads/
    echo          and tick "Add python.exe to PATH" during setup.
    goto :fail
)
echo  Using Python:
%PY% --version

REM ---- create the virtual environment -------------------------------------
if not exist ".venv\Scripts\python.exe" (
    echo.
    echo  [1/3] Creating virtual environment in .venv ...
    %PY% -m venv .venv || goto :fail
) else (
    echo.
    echo  [1/3] Virtual environment already exists - reusing .venv
)
set "VPY=.venv\Scripts\python.exe"

echo.
echo  [2/3] Upgrading pip ...
"%VPY%" -m pip install --upgrade pip || goto :fail

echo.
echo  [3/3] Installing StrikeCast packages (this can take a few minutes) ...
"%VPY%" -m pip install --upgrade ^
    "fastapi>=0.115" ^
    "uvicorn[standard]>=0.30" ^
    "pydantic>=2.8" ^
    "pydantic-settings>=2.4" ^
    "httpx>=0.27" ^
    "truststore>=0.9" ^
    "sqlalchemy>=2.0" ^
    "apscheduler>=3.10,<4" ^
    "pyyaml>=6.0" ^
    "python-dotenv>=1.0" ^
    "pandas>=2.2" ^
    "numpy>=1.26" ^
    "scipy>=1.13" ^
    "ephem>=4.1.5" ^
    "scikit-learn>=1.5" ^
    "joblib>=1.3" ^
    "tzdata>=2024.1" ^
    "streamlit>=1.38" ^
    "streamlit-folium>=0.22" ^
    "folium>=0.17" ^
    "plotly>=5.23" ^
    "pytest>=8.3" ^
    "pytest-asyncio>=0.24" || goto :fail

echo.
echo  Checking the installation ...
"%VPY%" -c "import importlib.util as u, sys; m=[n for n in 'fastapi uvicorn pydantic pydantic_settings httpx truststore sqlalchemy apscheduler yaml dotenv pandas numpy scipy ephem sklearn joblib streamlit streamlit_folium folium plotly pytest'.split() if u.find_spec(n) is None]; print('  Missing: '+', '.join(m) if m else '  All packages installed OK'); sys.exit(1 if m else 0)" || goto :fail

echo.
echo  ===========================================
echo    Install complete. Run start.bat to launch.
echo  ===========================================
echo.
if /i not "%~1"=="/nopause" pause
exit /b 0

:fail
echo.
echo  [ERROR] Installation failed - see the messages above.
echo.
if /i not "%~1"=="/nopause" pause
exit /b 1
