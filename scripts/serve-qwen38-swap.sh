#!/usr/bin/env bash
# llama-swap foreground wrapper for the qwen3.8-27b-skinny flex-pool entry.
#
# The native launcher (serve-qwen38-native.sh) backgrounds the server and
# exits, which llama-swap reads as upstream death. This wrapper stays in the
# foreground: llama-swap TERMs it to evict, the trap forwards TERM to vLLM,
# and the wrapper only exits after the GPUs are actually released — because
# llama-swap's only ordering guarantee to the NEXT model is "child exited",
# and vLLM's TP2/NCCL teardown can outlast process exit.
#
# Env block and vLLM flags MIRROR serve-qwen38-native.sh + the proven manual
# launch (REASONING_EFFORT=low TP=2 GMU=0.95 MML=204800 DECODE_PARTITION=1024).
# Keep the two scripts in LOCKSTEP when the native one changes.
#
# Usage: serve-qwen38-swap.sh <port>   (llama-swap passes ${PORT})
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PORT="${1:?usage: $0 <port> [served-model-name]}"
# Served name = llama-swap entry key (vLLM enforces the match). 2nd arg lets one
# wrapper back several entries (s359: qwen3.8-27b-skinny-mns4 = MNS=4 MBT=8192 via env:).
SERVED_NAME="${2:-${SERVED_NAME:-qwen3.8-27b-skinny}}"
CKPT="${CKPT:-/mnt/models/RadixArk-Qwen3.8-27B-NVFP4}"
PY="$REPO_ROOT/.venv-sm70/bin/python"
GPUS="${CUDA_VISIBLE_DEVICES:-2,3}"
K=7

# A just-evicted model's VRAM teardown can outlast its process exit; booting
# over occupied cards yields a fractional-speed server instead of a failure
# (native script check #1). Bounded wait, then proceed regardless — llama-swap
# health-checking catches a genuinely wedged boot.
for _ in $(seq 1 45); do
  USED=$(nvidia-smi -i "$GPUS" --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)
  [ "${USED:-0}" -lt 200 ] && break
  sleep 2
done

# NUMA pin is part of the launch config (native script: unpinned boots re-roll
# thread/page placement for a ±3% round-time lottery).
NUMA_PREFIX=""
command -v numactl >/dev/null && NUMA_PREFIX="numactl --cpunodebind=0 --membind=0"

# CUDA-graph capture sizes are MNS-aware (s358): every sequence steps K+1 tokens
# under MTP, so MNS=1 keeps the proven [8,16] and MNS=4 captures [8,16,24,32]
# instead of falling to eager above 2 streams. LOCKSTEP with serve-qwen38-native.sh.
K1=$((K + 1)); CAPS=$(seq -s, "$K1" "$K1" $((K1 * ( ${MNS:-1} > 2 ? ${MNS:-1} : 2 ))))
export CUDA_VISIBLE_DEVICES="$GPUS"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.8}"
export PATH="$REPO_ROOT/.venv-sm70/bin:$CUDA_HOME/bin:$PATH"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-/mnt/bulk/sigsec-scratch/.cache/torch_extensions}"
# s358 (2026-09-14): the JIT kernels (skinny_nvfp4_v11, flash_qla_sm70_gdn_strided)
# rebuild EVERY boot and torch's CUDAContextLight.h pulls cusparse/cublas/cusolver
# headers. The system 12.8 tree now carries only nvcc (cuda-nvcc-12-8, restored after
# the 09-12 root-disk purge) — the headers come from the venv's pip CUDA wheels.
# LOCKSTEP with serve-qwen38-native.sh.
export CPATH="$REPO_ROOT/.venv-sm70/lib/python3.12/site-packages/nvidia/cusparse/include:$REPO_ROOT/.venv-sm70/lib/python3.12/site-packages/nvidia/cublas/include:$REPO_ROOT/.venv-sm70/lib/python3.12/site-packages/nvidia/cusolver/include:$REPO_ROOT/.venv-sm70/lib/python3.12/site-packages/nvidia/curand/include:$REPO_ROOT/.venv-sm70/lib/python3.12/site-packages/nvidia/cufft/include${CPATH:+:$CPATH}"
export TORCH_CUDA_ARCH_LIST=7.0
export VLLM_SM70_NVFP4_TURBOMIND=0
export VLLM_SM70_QUANT_BACKEND=marlin
export VLLM_1CAT_ENABLE_SM70_MTP_DEFAULTS=1
export VLLM_SKINNY_NVFP4=1
export VLLM_SKINNY_QPN=1
export VLLM_SKINNY_QPN2=1
export VLLM_SKINNY_LMHEAD=1
export VLLM_SKINNY_LMHEAD_NATIVE=1
export VLLM_SKINNY_DROP_CT=1
export VLLM_SKINNY_NVFP4_SRC="$REPO_ROOT/kernels/skinny_kernels.cu"
export VLLM_SM70_MTP_DYNAMIC_DRAFT_VOCAB_DEFAULT=0
export VLLM_SM70_GDN_CHAIN_SPEC_FAST_BUILD=1
export VLLM_SM70_QPN8_MT2=1
export VLLM_FLASH_V100_DECODE_PARTITION_SIZE="${DECODE_PARTITION:-1024}"
# Telemetry opt-out: vLLM usage stats (stats.vllm.ai, boot + 600 s heartbeat) and
# HF hub user-agent telemetry. Belt-and-braces with ~/.config/vllm/do_not_track.
export VLLM_NO_USAGE_STATS=1
export DO_NOT_TRACK=1
export HF_HUB_DISABLE_TELEMETRY=1
# Checkpoint is a local dir with no auto_map remote code; proven boot 2026-09-01 never touched huggingface.co.
export HF_HUB_OFFLINE=1

$NUMA_PREFIX "$PY" -m vllm.entrypoints.openai.api_server \
  --model "$CKPT" \
  --served-model-name "$SERVED_NAME" \
  --trust-remote-code \
  --dtype float16 \
  --attention-backend FLASH_ATTN_V100 \
  --tensor-parallel-size "${TP:-2}" \
  --gpu-memory-utilization "${GMU:-0.95}" \
  --max-model-len "${MML:-204800}" \
  --max-num-seqs "${MNS:-1}" \
  --max-num-batched-tokens "${MBT:-4096}" \
  --limit-mm-per-prompt '{"image":0,"video":0}' \
  --default-chat-template-kwargs "{\"enable_thinking\":${THINKING:-true},\"reasoning_effort\":\"${REASONING_EFFORT:-low}\"}" \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  --compilation-config "{\"cudagraph_capture_sizes\":[$CAPS]}" \
  --speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":$K,\"draft_sample_method\":\"greedy\",\"use_local_argmax_reduction\":true}" \
  --host 127.0.0.1 --port "$PORT" &
CHILD=$!

trap 'kill -TERM "$CHILD" 2>/dev/null' TERM INT
wait "$CHILD"
RC=$?
# If wait was interrupted by the trap, wait again for the real exit.
wait "$CHILD" 2>/dev/null

# Hold llama-swap until VRAM is actually free for the next tenant.
for _ in $(seq 1 30); do
  USED=$(nvidia-smi -i "$GPUS" --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)
  [ "${USED:-0}" -lt 200 ] && break
  sleep 1
done
exit "$RC"
