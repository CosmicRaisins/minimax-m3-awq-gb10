#!/bin/bash
set -euo pipefail

MOD_DIR="$(dirname "$0")"
SITE_PACKAGES="${SITE_PACKAGES:-/usr/local/lib/python3.12/dist-packages}"
TARGET="${SITE_PACKAGES}/vllm/tool_parsers/minimax_m3_tool_parser.py"

echo "[fix-minimax-m3-tool-parser] Installing Python MiniMax M3 parser fallback"
install -m 0644 "${MOD_DIR}/minimax_m3_tool_parser.py" "${TARGET}"
python3 -m py_compile "${TARGET}"
echo "[fix-minimax-m3-tool-parser] Patched ${TARGET}"
