# StrikeCast launcher for Windows PowerShell.
#   .\run.ps1            -> dashboard (http://localhost:8501)
#   .\run.ps1 api        -> REST API  (http://127.0.0.1:8000/docs)
#   .\run.ps1 test       -> run the test suite
param([string]$target = "dashboard")
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (-not (Test-Path ".venv\Scripts\python.exe")) {
    Write-Host "Creating local virtual environment (.venv) and installing packages..."
    python -m venv .venv
    .\.venv\Scripts\python.exe -m pip install --upgrade pip
    .\.venv\Scripts\python.exe -m pip install -r requirements.txt
}

switch ($target) {
    "api"  { .\.venv\Scripts\python.exe -m uvicorn app.main:app --port 8000 }
    "test" { .\.venv\Scripts\python.exe -m pytest -q }
    default { .\.venv\Scripts\python.exe -m streamlit run dashboard/streamlit_app.py }
}
