#!/usr/bin/env bash
# Activate virtual environment and run benchmark.py
source .venv/Scripts/activate 2>/dev/null || source .venv/bin/activate
python tools/benchmark.py "$@"
