# MiniMax-M3-AWQ on DGX Spark / GB10 (sm_121) — vLLM patches + recipe

Run [`cyankiwi/MiniMax-M3-AWQ-INT4`](https://huggingface.co/cyankiwi/MiniMax-M3-AWQ-INT4) on 4× GB10 (NVIDIA DGX Spark, sm_121 aarch64) under vLLM with:

- **fp8 KV cache** — ~2× usable context / concurrency, throughput-neutral
- **Correct dense/long-context output** — fixes the AWQ quantized-indexer corruption (garbled / context-bleed on long, instruction-heavy prompts)
- **EAGLE3 speculative decoding** (k=2)
- **Adaptive reasoning** — thinks only when it helps, clean streaming (no `<mm:think>` leak)

Four small vLLM patches (`mods/`) + a serve recipe (`recipe.yaml`).

## Benchmark (4× GB10, TP=4, EAGLE3 k=2, llama-benchy)

| | bf16 KV | **fp8 KV (this setup)** |
|---|---|---|
| Decode @ 32k ctx | 28.6 tok/s | **29.4 tok/s** |
| Prefill | ~1.6k tok/s | ~1.7k tok/s |
| KV cache @ 262k ctx | ~0.69M tokens | **1.38M tokens** |
| Max concurrency @ 262k | 2.6× | **5.27×** |

Decode is throughput-neutral (fp8 ≈ bf16); fp8's win is capacity. Decode across depth: ~32 / 30 / 29 / 26 tok/s at 0 / 8k / 32k / 64k.

## Recipe (`vllm serve`)

```bash
vllm serve cyankiwi/MiniMax-M3-AWQ-INT4 \
  --served-model-name minimax-m3-awq --trust-remote-code --block-size 128 \
  --attention-backend TRITON_ATTN --kv-cache-dtype fp8 --language-model-only \
  -tp 4 --distributed-executor-backend ray --gpu-memory-utilization 0.90 \
  --max-model-len 262144 --max-num-batched-tokens 8192 --max-num-seqs 4 \
  --enable-prefix-caching --enforce-eager \
  --reasoning-parser minimax_m3 --enable-auto-tool-choice --tool-call-parser minimax_m3 \
  --speculative-config '{"method":"eagle3","model":"Inferact/MiniMax-M3-EAGLE3","num_speculative_tokens":2,"attention_backend":"TRITON_ATTN"}'
```

For adaptive reasoning, requests pass `chat_template_kwargs: {"thinking_mode": "adaptive"}` (or `"enabled"`/`"disabled"`).

## Patches (`mods/` — each `run.sh` patches the installed vLLM)

- **fix-minimax-m3-compressed-tensors** — un-fuses the bf16 MSA "lightning indexer" out of the INT4-quantized qkv projection. The AWQ checkpoint keeps the indexer in bf16 while q/k/v are INT4; the fused path quantized the indexer too → mis-selected tokens → garbled / context-bleed output under long/dense context. Ported from [toncao/vllm `minimax-m3-compressed-tensors`](https://github.com/toncao/vllm/tree/minimax-m3-compressed-tensors).
- **fix-minimax-m3-reasoning-45718** — streaming reasoning parser ([vLLM PR #45718](https://github.com/vllm-project/vllm/pull/45718)) + chat-template tweak, so `thinking_mode: adaptive` streams reasoning into `reasoning_content` without leaking `<mm:think>` into content.
- **fix-minimax-m3-tool-parser** — MiniMax-M3 tool-call parser.
- **fix-nccl-2.30.4** — swaps bundled libnccl for 2.30.4 on every rank (fixes a shm_broadcast wedge-under-load on multi-node Ray).

## Requirements / use

1. **vLLM with the fp8 sparse-GQA kernel** ([PR #45744](https://github.com/vllm-project/vllm/pull/45744), merged upstream). On GB10/sm_121 build vLLM for `sm_121a` (aarch64); on x86/datacenter a recent build already has it.
2. Apply the four mods to the vLLM install (run each `run.sh`).
3. Serve with the recipe above.

Notes: fp8 KV requires the EAGLE3 draft on a fp8-capable attention backend (`TRITON_ATTN`, **not** `FLASH_ATTN`). Tested on vLLM `0.22.1` + CUDA 13, 4× GB10.

Patches derive from vLLM (Apache-2.0); credit to toncao (indexer) and the PR #45718 author (reasoning).
