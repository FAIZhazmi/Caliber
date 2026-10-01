@echo off
setlocal
cd /d "%~dp0"
set "PYTHONPATH=.vendor;src"
python -m streamlit run dashboard\\app.py --server.address 127.0.0.1 --server.port 8502 --server.headless true --browser.gatherUsageStats false

