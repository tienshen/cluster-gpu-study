@echo off
REM Activate virtual environment and run benchmark.py
call .venv\Scripts\activate.bat
python tools\benchmark.py %*
