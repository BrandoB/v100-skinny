#!/usr/bin/env bash
# Stage 0 window C: cross-pair all-reduce latency (Volta roadmap Task 0.3). Brandon's go only; never test-run (`bash -n`).
# Precondition (Brandon): the WHOLE stack is down —
#   ~/jarvis/scripts/stack.sh down && systemctl --user stop jarvis-phone-bridge jarvis-phone-tts
# Then: cd /mnt/bulk/sigsec-scratch/v100-skinny && nohup bash scripts/stage0_windowC.sh > results/stage0/windowC.out 2>&1 &
# Refuses unless all 4 GPUs are empty. Every exit (incl. Ctrl-C / kill -TERM) runs stack.sh up + phone-tts; the phone
# bridge is started only if it is enabled (leave-as-found). Each torchrun is bounded by `timeout 300` (TERM, never KILL).
set -u
cd /mnt/bulk/sigsec-scratch/v100-skinny || exit 1
export XDG_RUNTIME_DIR=${XDG_RUNTIME_DIR:-/run/user/$(id -u)}
RES=results/stage0; mkdir -p "$RES"; LOGF=$RES/windowC.log; OUT=$RES/ar_latency.txt; SMI=; CH=
log() { echo "$(date +%H:%M:%S) $*" | tee -a "$LOGF"; }
N=$(( 10#$(date +%H) * 60 + 10#$(date +%M) ))
[ "$N" -ge 480 ] && [ "$N" -le 1230 ] || { log "REFUSE: outside 08:00-20:30"; exit 1; }
APPS=$(nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name --format=csv,noheader)
[ -z "$APPS" ] || { log "REFUSE: GPUs not empty — stop the stack first. Holders: $(echo "$APPS" | tr '\n' ';')"; exit 1; }
restore() {
  local rc=$?
  trap '' TERM INT
  [ -n "$CH" ] && kill -0 "$CH" 2>/dev/null && { kill -TERM "$CH" 2>/dev/null; for _ in $(seq 1 30); do kill -0 "$CH" 2>/dev/null || break; sleep 2; done; }
  [ -n "$SMI" ] && kill "$SMI" 2>/dev/null
  /home/brand/jarvis/scripts/stack.sh up >> "$LOGF" 2>&1
  systemctl --user start jarvis-phone-tts
  [ "$(systemctl --user is-enabled jarvis-phone-bridge 2>/dev/null)" = enabled ] && systemctl --user start jarvis-phone-bridge
  log "RESTORED (rc $rc) — check: curl -sk https://localhost:8090/health ; curl -s 127.0.0.1:8082/running"
}
trap restore EXIT
trap 'log "TERM received"; exit 143' TERM
trap 'log "INT received"; exit 130' INT
log "WINDOW C START $(date -Is)"
nvidia-smi --query-gpu=timestamp,index,clocks.sm,clocks.mem,power.draw,temperature.gpu,clocks_event_reasons.active --format=csv -lms 500 > "$RES/windowC.smi.csv" & SMI=$!
for G in 2,3 0,1 1,2 0,1,2,3; do
  NP=$(echo "$G" | tr , '\n' | wc -l)
  CUDA_VISIBLE_DEVICES=$G NCCL_DEBUG=INFO timeout -s TERM 300 .venv-sm70/bin/torchrun --nproc_per_node "$NP" benchmarks/ar_latency.py \
    > >(tee -a "$OUT") 2> "$RES/ar_latency.$G.nccl.log" & CH=$!
  wait "$CH" || log "torchrun $G rc=$?"; CH=
  log "$G transport: $(grep -ohE 'via (P2P/IPC|P2P/direct pointer|SHM|NET)[^ ]*' "$RES/ar_latency.$G.nccl.log" | sort | uniq -c | tr '\n' ' ')"
done
log "WINDOW C arms complete"
