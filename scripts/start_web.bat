@echo off
setlocal
cd /d "%~dp0.."

set "PY=D:\miniconda\envs\agent1\python.exe"
if not exist "%PY%" set "PY=python"

echo Starting Ops Diagnostic Agent...
start "OpsAgent API" "%PY%" -m uvicorn app.api:app --host 127.0.0.1 --port 8000
timeout /t 3 /nobreak >nul
start "OpsAgent Web" "%PY%" -m streamlit run ui\streamlit_app.py --server.address 127.0.0.1 --server.port 8501 --server.headless true
timeout /t 6 /nobreak >nul
start "" http://localhost:8501

echo API: http://localhost:8000/docs
echo UI : http://localhost:8501
endlocal