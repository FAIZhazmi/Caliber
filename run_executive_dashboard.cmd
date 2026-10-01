@echo off
setlocal
cd /d "%~dp0"
set "PYTHONPATH=.vendor;src"
python -m streamlit run dashboard\\executive_app.py --server.address 127.0.0.1 --server.port 8501 --server.headless true --browser.gatherUsageStats false

