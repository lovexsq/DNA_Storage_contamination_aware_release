$ErrorActionPreference = "Stop"

if (-not (Test-Path ".\.venv\Scripts\python.exe")) {
    Write-Host "未找到虚拟环境。请先运行 setup_env.ps1。"
    exit 1
}

.\.venv\Scripts\python.exe evaluate_existing_checkpoints.py --test-clusters 20 --load-buffer 50 --test-contamination-rates 0.00 0.30 --results-dir results/quick_check
