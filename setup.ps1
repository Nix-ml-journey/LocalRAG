# Windows PowerShell: .\setup.ps1
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -U pip
# CPU-only torch keeps the install small; the GPU is used by Ollama only
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
ollama pull gemma3:12b   # must match llm.model in config.yaml
Write-Host "Done. Activate with .\.venv\Scripts\Activate.ps1, then: python cli.py ingest"
