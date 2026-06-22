#!/bin/bash
set -euo pipefail
# fix-minimax-m3-compressed-tensors: proper MiniMax-M3 compressed-tensors (AWQ INT4)
# support, ported from toncao/vllm @ minimax-m3-compressed-tensors (base upstream
# a7fdfeef, 2026-06-16). Replaces the fix-m3-swiglu-clamp band-aid. Two parts:
#  (1) nvidia/model.py: UN-FUSE the sparse-attention lightning indexer out of the
#      INT4-quantized qkv_proj into standalone bf16 ReplicatedLinear projections.
#      cyankiwi/MiniMax-M3-AWQ-INT4 keeps index_q/index_k in bf16 while q/k/v are
#      INT4; the fused MinimaxM3QKVParallelLinearWithIndexer path quantized the
#      indexer too -> corrupted MSA token selection -> memory-bleed/garbage under
#      long/dense context (clean on short prompts). This is the root-cause fix.
#  (2) clamp plumbing: thread swiglu_limit/alpha/beta from the layer into the
#      int4/int8 wna16 MoE quant configs so SWIGLUOAI_UNINTERLEAVE receives the
#      REAL checkpoint clamp (no hardcoded 7.0/1.702/1.0 backfill).
DIFF="$PWD/toncao-m3.diff"   # run.sh is invoked via `cd $container_dest && ./run.sh`
SITE_PACKAGES="${SITE_PACKAGES:-/usr/local/lib/python3.12/dist-packages}"
cd "$SITE_PACKAGES"
if grep -q "index_q_proj = ReplicatedLinear" vllm/models/minimax_m3/nvidia/model.py 2>/dev/null; then
  echo "[fix-minimax-m3-compressed-tensors] already applied; skipping"
  exit 0
fi
patch -p1 --batch < "$DIFF"
echo "[fix-minimax-m3-compressed-tensors] applied (indexer un-fuse + clamp plumbing)"
