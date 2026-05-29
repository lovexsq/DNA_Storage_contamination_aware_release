$ErrorActionPreference = "Stop"

if (-not (Test-Path ".\.venv\Scripts\python.exe")) {
    Write-Host "未找到虚拟环境。请先运行 setup_env.ps1。"
    exit 1
}

.\.venv\Scripts\python.exe run_contamination_aware_experiment.py --train-clusters 1000 --val-clusters 150 --test-clusters 300 --load-buffer 1000 --test-contamination-rates 0.00 0.05 0.10 0.20 0.30 --results-dir results/contamination_aware_train1000_test300_with_clean --checkpoint-dir checkpoints/contamination_aware_train1000_test300_with_clean
