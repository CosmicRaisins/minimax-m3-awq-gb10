#!/usr/bin/env bash
set -euo pipefail

MOD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VLLM_DIR="$(
  python3 - <<'PY'
import pathlib
import vllm

print(pathlib.Path(vllm.__file__).resolve().parent)
PY
)"
TARGET="${VLLM_DIR}/reasoning/minimax_m3_reasoning_parser.py"

install -m 0644 "${MOD_DIR}/minimax_m3_reasoning_parser.py" "${TARGET}"
python3 -m py_compile "${TARGET}"
python3 "${MOD_DIR}/patch_chat_template.py"
