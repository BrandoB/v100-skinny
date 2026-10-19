#!/usr/bin/env bash
# llama-swap foreground wrapper: STOCK 1Cat-vLLM 1.5.1 + DFlash2 for the qwen3.8-27b-skinny entry (2026-10-18).
# Same contract as serve-qwen38-swap.sh (stays in the foreground, forwards TERM, exits only after VRAM is released),
# but the launch is the one Stage 0 window D measured (results/stage0/summary-windowD.md):
#   .venv-1cat151 (no fork_patches), RadixArk Qwen3.8-27B-NVFP4-BF16-LMHead, fp8_e4m3 KV, GMU 0.85, the wheel's own
#   release-profile argv (python -m vllm.sm70_profiles argv qwen38_27b_nvfp4_dflash2) + TP2/port/name/limits overrides.
# Environment is ALLOWLISTED (env -i): the skinny env (VLLM_SM70_QUANT_BACKEND=marlin, NVFP4_TURBOMIND=0, VLLM_SKINNY_*)
# makes 1.5.1 refuse this checkpoint on SM70. Production semantics kept from the skinny entry: thinking ON with
# reasoning_effort low by default (per-request adjustable), MNS 4, MBT 8192, 204800 context, every served name passed.
# Rollback = point the llama-swap entry back at serve-qwen38-swap.sh (kept as entry qwen3.8-27b-skinny-legacy).
# Usage: serve-1cat-stock-swap.sh <port> <served-model-name> [alias ...]
set -uo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PORT="${1:?usage: $0 <port> <served-model-name> [alias ...]}"; shift
[ "$#" -gt 0 ] || { echo "usage: $0 <port> <served-model-name> [alias ...]" >&2; exit 2; }
SERVED_NAMES=("$@")
ENV_PREFIX="${ENV_PREFIX:-/mnt/bulk/sigsec-scratch/.venv-1cat151}"
PY="$ENV_PREFIX/bin/python"
CKPT="${CKPT:-/mnt/models/RadixArk-Qwen3.8-27B-NVFP4-BF16-LMHead}"
DRAFT="${DRAFT:-/mnt/models/Qwen3.8-27B-DFlash2}"
GPUS="${CUDA_VISIBLE_DEVICES:-2,3}"; K="${K:-7}"
[ -x "$PY" ] && [ -d "$CKPT" ] && [ -d "$DRAFT" ] || { echo "ERROR: env/checkpoint/draft missing" >&2; exit 2; }

for _ in $(seq 1 45); do   # bounded wait for the previous tenant's VRAM (same as serve-qwen38-swap.sh)
  USED=$(nvidia-smi -i "$GPUS" --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)
  [ "${USED:-0}" -lt 200 ] && break
  sleep 2
done

mapfile -t BASE < <(cd /tmp && CUDA_VISIBLE_DEVICES= "$PY" -I -m vllm.sm70_profiles argv qwen38_27b_nvfp4_dflash2 --argv-lines --draft "$DRAFT")
[ "${#BASE[@]}" -gt 20 ] || { echo "ERROR: profile argv unreadable (${#BASE[@]} lines)" >&2; exit 2; }
SPEC="{\"method\":\"dflash\",\"model\":\"$DRAFT\",\"num_speculative_tokens\":$K,\"kv_cache_dtype\":\"auto\",\"attention_backend\":\"FLASH_ATTN_V100\",\"draft_sample_method\":\"probabilistic\",\"enforce_eager\":false}"
ARGS=("${BASE[@]}" --port "$PORT" --tensor-parallel-size "${TP:-2}" --served-model-name "${SERVED_NAMES[@]}"
  --max-model-len "${MML:-204800}" --gpu-memory-utilization "${GMU:-0.85}" --kv-cache-dtype "${KV:-fp8_e4m3}"
  --max-num-seqs "${MNS:-4}" --max-num-batched-tokens "${MBT:-8192}"
  --default-chat-template-kwargs "{\"enable_thinking\":${THINKING:-true},\"reasoning_effort\":\"${REASONING_EFFORT:-low}\"}"
  --speculative-config "$SPEC")
NUMA=(); command -v numactl >/dev/null && NUMA=(numactl --cpunodebind=0 --membind=0)

env -i HOME="$HOME" USER="${USER:-brand}" PATH="$ENV_PREFIX/bin:/usr/local/cuda-12.8/bin:/usr/bin:/bin" \
  CUDA_HOME=/usr/local/cuda-12.8 CUDA_VISIBLE_DEVICES="$GPUS" HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 \
  DO_NOT_TRACK=1 VLLM_NO_USAGE_STATS=1 TORCH_EXTENSIONS_DIR=/mnt/bulk/sigsec-scratch/.cache/torch_extensions-1cat151 \
  "${NUMA[@]}" "$PY" -m vllm.entrypoints.openai.api_server --model "$CKPT" "${ARGS[@]}" &
CHILD=$!
trap 'kill -TERM "$CHILD" 2>/dev/null' TERM INT
wait "$CHILD"; RC=$?
wait "$CHILD" 2>/dev/null
for _ in $(seq 1 30); do   # hold llama-swap until VRAM is free for the next tenant
  USED=$(nvidia-smi -i "$GPUS" --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)
  [ "${USED:-0}" -lt 200 ] && break
  sleep 1
done
exit "$RC"
