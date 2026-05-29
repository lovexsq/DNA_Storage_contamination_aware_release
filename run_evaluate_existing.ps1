$ErrorActionPreference = "Stop"

if (-not (Test-Path ".\.venv\Scripts\python.exe")) {
    Write-Host "未找到虚拟环境。请先运行 setup_env.ps1。"
    exit 1
}

.\.venv\Scripts\python.exe evaluate_existing_checkpoints.py
