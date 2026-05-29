#!/usr/bin/env bash
set -euo pipefail

echo "正在创建本地虚拟环境：.venv"
python3 -m venv .venv

PYTHON=".venv/bin/python"

echo "正在升级 pip"
"$PYTHON" -m pip install --upgrade pip

echo "正在安装 CPU 版 PyTorch"
"$PYTHON" -m pip install torch==2.12.0+cpu --index-url https://download.pytorch.org/whl/cpu

echo "正在安装其余依赖"
"$PYTHON" -m pip install -r requirements.txt

echo "正在运行发布包完整性检查"
"$PYTHON" verify_release.py

echo "环境已准备完成。"
