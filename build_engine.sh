#!/usr/bin/env bash
# Activate virtual environment and run build_engine.py
source .venv/Scripts/activate 2>/dev/null || source .venv/bin/activate
python tools/build_engine.py "$@"
