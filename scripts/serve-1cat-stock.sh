#!/usr/bin/env bash
# Stage 0 comparison lane: STOCK 1Cat-vLLM 1.5.1 (no fork_patches) on the manual :8096 slot.
#   ARM=mtp|dflash2 scripts/serve-1cat-stock.sh <checkpoint>
# Base argv = the wheel's own release profile (`python -m vllm.sm70_profiles argv qwen38_27b_nvfp4_dflash2`),
# then TP2/port/name/memory overrides (argparse: last flag wins). Returns after READY with the server setsid'd,
# like serve-qwen38-native.sh, and writes the same $REPO_ROOT/serve.pid (UPS shed finds the manual lane there).
# The environment is ALLOWLISTED (env -i): native.sh's VLLM_SM70_QUANT_BACKEND=marlin / NVFP4_TURBOMIND=0 would
# make 1.5.1 refuse this checkpoint on SM70. Boot gates (each can fail): env audit, resolved KV dtype, served
# spec method+depth, acceleration report, warm completion, known-answer correctness. Never test-run outside a
# Stage 0 window (Brandon's go).
# Overridable: ENV_PREFIX ARM K PORT TP GMU MML KV DRAFT LOG PIDFILE CUDA_VISIBLE_DEVICES DRAFT_VOCAB_DEFAULT
set -u
REPO_ROOT=$(cd "$(dirname "$0")/.." && pwd)
CKPT=${1:?usage: ARM=mtp|dflash2 $0 <checkpoint>}
ENV_PREFIX=${ENV_PREFIX:-/mnt/bulk/sigsec-scratch/.venv-1cat151}
PY=$ENV_PREFIX/bin/python
ARM=${ARM:?ARM=mtp|dflash2}; K=${K:-7}; PORT=${PORT:-8096}; TP=${TP:-2}; GMU=${GMU:-0.85}; MML=${MML:-204800}
KV=${KV:-fp8_e4m3}; DRAFT=${DRAFT:-/mnt/models/Qwen3.8-27B-DFlash2}; GPUS=${CUDA_VISIBLE_DEVICES:-2,3}
LOG=${LOG:-$REPO_ROOT/serve-1cat-stock.log}; PIDFILE=${PIDFILE:-$REPO_ROOT/serve.pid}
[ -x "$PY" ] || { echo "ERROR: no env at $ENV_PREFIX" >&2; exit 2; }
[ -d "$CKPT" ] || { echo "ERROR: no checkpoint $CKPT" >&2; exit 2; }
case "$ARM" in
  dflash2) SPEC="{\"method\":\"dflash\",\"model\":\"$DRAFT\",\"num_speculative_tokens\":$K,\"kv_cache_dtype\":\"auto\",\"attention_backend\":\"FLASH_ATTN_V100\",\"draft_sample_method\":\"probabilistic\",\"enforce_eager\":false}"; METHOD=dflash ;;
  mtp)     SPEC="{\"method\":\"mtp\",\"num_speculative_tokens\":$K,\"draft_sample_method\":\"greedy\",\"use_local_argmax_reduction\":true}"; METHOD=mtp ;;
  *) echo "ERROR: ARM must be mtp or dflash2" >&2; exit 2 ;;
esac
USED=$(nvidia-smi -i "$GPUS" --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)
[[ "$USED" =~ ^[0-9]+$ ]] && [ "$USED" -lt 200 ] || { echo "ERROR: GPUs $GPUS not free (${USED:-?} MiB)" >&2; exit 1; }

mapfile -t BASE < <(cd /tmp && CUDA_VISIBLE_DEVICES= "$PY" -I -m vllm.sm70_profiles argv qwen38_27b_nvfp4_dflash2 --argv-lines --draft "$DRAFT")
[ "${#BASE[@]}" -gt 20 ] || { echo "ERROR: profile argv unreadable (${#BASE[@]} lines)" >&2; exit 2; }
ARGS=("${BASE[@]}" --port "$PORT" --tensor-parallel-size "$TP" --served-model-name qwen3.8-27b
  --max-model-len "$MML" --gpu-memory-utilization "$GMU" --kv-cache-dtype "$KV" --speculative-config "$SPEC")
EXTRA=(); [ -n "${DRAFT_VOCAB_DEFAULT:-}" ] && EXTRA=(VLLM_SM70_MTP_DYNAMIC_DRAFT_VOCAB_DEFAULT="$DRAFT_VOCAB_DEFAULT")
NUMA=(); command -v numactl >/dev/null && NUMA=(numactl --cpunodebind=0 --membind=0)   # same pin as the skinny lanes

{ echo "# serve-1cat-stock $(date -Is) ARM=$ARM K=$K TP=$TP GMU=$GMU MML=$MML KV=$KV ckpt=$CKPT env=$ENV_PREFIX ${EXTRA[*]}"
  echo "# argv: ${ARGS[*]}"; } > "$LOG"
cleanup_on_fail() {  # TERM the whole setsid group (PID == PGID): EngineCore/worker stragglers too, even if the leader died
  [ -s "$PIDFILE" ] || return 0
  local pid; pid=$(cat "$PIDFILE")
  echo "==> tearing down the server group (pid $pid)" >&2
  kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
  for _ in $(seq 1 60); do pgrep -g "$pid" >/dev/null 2>&1 || break; sleep 2; done
  rm -f "$PIDFILE"
}
trap 'cleanup_on_fail; exit 143' TERM; trap 'cleanup_on_fail; exit 130' INT
setsid env -i HOME="$HOME" USER="$USER" PATH="$ENV_PREFIX/bin:/usr/local/cuda-12.8/bin:/usr/bin:/bin" \
  CUDA_HOME=/usr/local/cuda-12.8 CUDA_VISIBLE_DEVICES="$GPUS" HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 \
  DO_NOT_TRACK=1 VLLM_NO_USAGE_STATS=1 TORCH_EXTENSIONS_DIR=/mnt/bulk/sigsec-scratch/.cache/torch_extensions-1cat151 \
  "${EXTRA[@]}" "${NUMA[@]}" "$PY" -m vllm.entrypoints.openai.api_server --model "$CKPT" "${ARGS[@]}" >> "$LOG" 2>&1 < /dev/null &
SERVER_PID=$!; echo "$SERVER_PID" > "$PIDFILE"
echo "==> waiting for the server (first 1.5.1 boot JIT-compiles TileLang kernels)"
UP=0
for _ in $(seq 1 600); do
  curl -sf -o /dev/null --max-time 2 "http://127.0.0.1:$PORT/v1/models" && { UP=1; break; }
  kill -0 "$SERVER_PID" 2>/dev/null || { echo "SERVER DIED — last lines:" >&2; tail -30 "$LOG" >&2; cleanup_on_fail; exit 1; }
  sleep 2
done
[ "$UP" = 1 ] || { echo "TIMEOUT waiting for the server" >&2; tail -30 "$LOG" >&2; cleanup_on_fail; exit 1; }

FAIL=0
gate() { if [ "$2" = ok ]; then echo "    PASS  $1"; else echo "    FAIL  $1 — $3"; FAIL=1; fi; }
echo "==> boot gate (stock 1.5.1, ARM=$ARM)"
BAD=$(for p in $(pgrep -s "$(ps -o sid= "$SERVER_PID" | tr -d ' ')"); do tr '\0' '\n' < /proc/$p/environ 2>/dev/null; done \
  | grep -E '^(VLLM_SKINNY_|VLLM_SM70_QUANT_BACKEND=|VLLM_SM70_NVFP4_TURBOMIND=)' | sort -u | tr '\n' ' ')
[ -z "$BAD" ] && gate "no skinny/fork env in the server's process tree" ok || gate "no skinny/fork env in the server's process tree" fail "$BAD"
# Gates read only the server's own lines (not this script's "# ..." header, which echoes the requested values).
# Formats from 1.5.1: VllmConfig.__str__ "kv_cache_dtype=<x>," (config/vllm.py:3345) and
# SpeculativeConfig.__repr__ "SpeculativeConfig(method='<m>', model=..., num_spec_tokens=<k>)" (speculative.py:1584).
SRV_LOG=$(grep -v '^#' "$LOG")
grep -q "kv_cache_dtype=$KV," <<< "$SRV_LOG" && gate "resolved kv_cache_dtype=$KV" ok \
  || gate "resolved kv_cache_dtype=$KV" fail "server logged: $(grep -oE 'kv_cache_dtype=[a-z0-9_]+' <<< "$SRV_LOG" | sort -u | tr '\n' ' ')"
grep -qE "SpeculativeConfig\(method='$METHOD', .*num_spec_tokens=$K\)" <<< "$SRV_LOG" \
  && gate "served speculative method=$METHOD depth=$K" ok \
  || gate "served speculative method=$METHOD depth=$K" fail "server logged: $(grep -oE 'SpeculativeConfig\([^)]*\)' <<< "$SRV_LOG" | head -1)"
curl -s -m 10 "http://127.0.0.1:$PORT/v1/sm70/acceleration" > "$LOG.accel.json"
ACC=$(ARM=$ARM python3 -I -c '
import json, os, sys
d = json.load(open(sys.argv[1])); p = d.get("paths", {})
need = ["batch_gemm", "compile_graph"] + (["dflash2_verifier"] if os.environ["ARM"] == "dflash2" else [])
bad = [n for n in need if not p.get(n, {}).get("enabled")]
print("ok" if not bad else "missing/disabled: " + ", ".join(n + " (" + str(p.get(n, {}).get("reason")) + ")" for n in bad))' "$LOG.accel.json" 2>&1)
[ "$ACC" = ok ] && gate "acceleration report (batch_gemm, compile_graph$([ "$ARM" = dflash2 ] && echo ', dflash2_verifier'))" ok \
  || gate "acceleration report" fail "$ACC"
WARM=$(curl -s -m 300 "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-27b","messages":[{"role":"user","content":"Say hello."}],"max_tokens":16,"temperature":0,"chat_template_kwargs":{"enable_thinking":false}}')
echo "$WARM" | python3 -I -c 'import json,sys; c=json.load(sys.stdin)["choices"][0]["message"]["content"]; sys.exit(0 if c and c.strip() else 1)' 2>/dev/null \
  && gate "warm request returned a valid completion" ok || gate "warm request returned a valid completion" fail "HTTP error or empty"
KA=$(PORT=$PORT python3 -I - <<'PY'
import json, os, urllib.request
cases = [("What is the capital of France? Answer with one word.", "paris"),
         ("What is 17 multiplied by 23? Answer with only the number.", "391"),
         ("Name the largest planet in our solar system. Answer with one word.", "jupiter")]
ok, out = 0, []
for q, a in cases:
    body = json.dumps({"model": "qwen3.8-27b", "messages": [{"role": "user", "content": q}], "max_tokens": 32,
                       "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{os.environ['PORT']}/v1/chat/completions", body, {"Content-Type": "application/json"})
    try:
        t = json.load(urllib.request.urlopen(req, timeout=120))["choices"][0]["message"]["content"] or ""
    except Exception as e:
        t = f"<error {type(e).__name__}>"
    ok += a in t.lower(); out.append(t.strip().replace("\n", " ")[:60])
print(f"{ok}/3 | " + " | ".join(out))
PY
)
echo "$KA" > "$LOG.knownanswer.txt"
[ "${KA%%/*}" -ge 2 ] 2>/dev/null && gate "known-answer correctness ($KA)" ok || gate "known-answer correctness" fail "$KA"
if [ "$FAIL" = 1 ]; then echo "==> GATE FAIL — tearing down" >&2; cleanup_on_fail; exit 1; fi
trap - TERM INT
echo "==> READY (pid $SERVER_PID, :$PORT, ARM=$ARM)"
