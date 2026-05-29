#!/usr/bin/env bash
set -euo pipefail

if [ ! -x ".venv/bin/python" ]; then
  echo "未找到虚拟环境。请先运行 setup_env.sh。"
  exit 1
fi

.venv/bin/python generate_enhanced_analysis.py
