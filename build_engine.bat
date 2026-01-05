@echo off
REM Activate virtual environment and run build_engine.py
call .venv\Scripts\activate.bat
python tools\build_engine.py %*
