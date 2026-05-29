$ErrorActionPreference = "Stop"

Write-Host "正在创建本地虚拟环境：.venv"
if (Get-Command py -ErrorAction SilentlyContinue) {
    py -3 -m venv .venv
} else {
    python -m venv .venv
}

$python = ".\.venv\Scripts\python.exe"

Write-Host "正在升级 pip"
& $python -m pip install --upgrade pip

Write-Host "正在安装 CPU 版 PyTorch"
& $python -m pip install torch==2.12.0+cpu --index-url https://download.pytorch.org/whl/cpu

Write-Host "正在安装其余依赖"
& $python -m pip install -r requirements.txt

Write-Host "正在运行发布包完整性检查"
& $python verify_release.py

Write-Host "环境已准备完成。"
