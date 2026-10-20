#!/usr/bin/env bash
# Linux: bash setup.sh
set -e
python3.12 -m venv .venv
.venv/bin/python -m pip install -U pip
.venv/bin/python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install -r requirements.txt
ollama pull gemma3:12b   # must match llm.model in config.yaml
echo "Done. source .venv/bin/activate, then: python cli.py ingest"
