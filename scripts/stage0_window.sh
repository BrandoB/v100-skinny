#!/usr/bin/env bash
# Stage 0 measurement window. Usage: scripts/stage0_window.sh A|B   (Brandon's go only)
# Volta roadmap Task 0.0 Step 3. Deliberately NEVER test-run: `bash -n` only, then Brandon's go.
# Run in the background so no tool timeout can strand the cards:
#   cd /mnt/bulk/sigsec-scratch/v100-skinny && nohup bash scripts/stage0_window.sh A > results/stage0/windowA.out 2>&1 &
# Abort: touch /tmp/stage0.abort (checked before every boot) or kill -TERM <pid> (exits, then restores).
# Restores llama-swap on every exit unless the stack was shed/restored under us (UPS) — then recovery
# belongs to restore-stable and the window keeps its hands off. Restore is verified (skinny "ready") and
# reported on Telegram. If GPUs 2,3 won't free, llama-swap is left DOWN with a CRITICAL alert rather than
# booted over held cards (the swap wrapper proceeds regardless after 90 s).
set -u
cd /mnt/bulk/sigsec-scratch/v100-skinny || exit 1
export XDG_RUNTIME_DIR=${XDG_RUNTIME_DIR:-/run/user/$(id -u)}   # systemctl --user from nohup/cron shells
WIN=${1:-}
case "$WIN" in
  A) EST_MIN=150 ;;   # 13 sweeps, baseline first and last
  B) EST_MIN=150 ;;   # 7 sweeps + 2 long-context arms
  D) EST_MIN=120 ;;   # skinny base -> stock 1Cat 1.5.1 MTP -> 1.5.1 DFlash2 -> skinny base, each + llama-benchy
  Q) EST_MIN=60 ;;    # production-switch quality gate on stock 1.5.1 + DFlash2 (benchmarks/quality_gate.py)
  *) echo "usage: $0 A|B|D|Q" >&2; exit 2 ;;
esac
# Night mode (Brandon's per-night go): NIGHT_OK=<YYYY-MM-DD the night starts> moves the deadline from 21:00 to
# 06:30 next morning. The fan guard below is what bounds noise at night (Brandon 2026-10-17: fans may not sit
# at 100% for more than 10-15 min).
FAN_LIMIT=${FAN_LIMIT:-95}; FAN_MAX_S=${FAN_MAX_S:-600}
STOCK_CKPT=/mnt/models/RadixArk-Qwen3.8-27B-NVFP4-BF16-LMHead
RES=results/stage0; mkdir -p "$RES"; LOGF=$RES/window.log; ABORT=/tmp/stage0.abort
SHED=0; STARTED=0; CH=; SMI=; TAG=; FANW=; MAIN=$$; WITH_BENCHY=0
log() { echo "$(date +%H:%M:%S) $*" | tee -a "$LOGF"; }
notify() { # notify SEVERITY MSG -> 0 only if Telegram delivered
  /home/brand/jarvis-venv/bin/python -c 'import sys; sys.path.insert(0, "/home/brand/jarvis"); import ops_send; r = ops_send.gateway_send(sys.argv[1], source="stage0", fmt="plain", severity=sys.argv[2]); sys.exit(0 if r else 1)' "$2" "$1"
}
now_min() { echo $(( 10#$(date +%H) * 60 + 10#$(date +%M) )); }
is_vllm() { [ -n "${1:-}" ] && tr '\0' ' ' < "/proc/$1/cmdline" 2>/dev/null | grep -q 'vllm.entrypoints'; }
run() { "$@" & CH=$!; wait "$CH"; local rc=$?; CH=; return $rc; }   # backgrounded so TERM interrupts the wait
foreign_on_23() {
  local out="" row U P I
  for row in $(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader | tr -d ' '); do
    U=${row%,*}; P=${row#*,}
    I=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | tr -d ' ' | awk -F, -v u="$U" '$2==u {print $1}')
    case "$I" in 2|3) grep -q 'llama-swap\.service' "/proc/$P/cgroup" 2>/dev/null || out="$out gpu$I:pid$P($(cat /proc/$P/comm 2>/dev/null))" ;; esac
  done
  echo "$out"
}
wait_free() { # wait_free SECS -> 0 once both GPUs 2,3 use < 200 MiB, 1 on timeout
  local t=0 U
  while :; do
    U=$(nvidia-smi -i 2,3 --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)
    [[ "$U" =~ ^[0-9]+$ ]] && [ "$U" -lt 200 ] && return 0
    [ "$t" -ge "$1" ] && return 1
    sleep 5; t=$((t + 5))
  done
}
stop_server() { # TERM the api_server, then its process group; bounded; 1 if it would not exit
  local P G; P=$(cat serve.pid 2>/dev/null)
  if is_vllm "$P"; then
    kill -TERM "$P" 2>/dev/null
    for _ in $(seq 1 24); do kill -0 "$P" 2>/dev/null || break; sleep 5; done
    if kill -0 "$P" 2>/dev/null; then
      G=$(ps -o pgid= "$P" 2>/dev/null | tr -d ' '); [ -n "$G" ] && kill -TERM "-$G" 2>/dev/null
      for _ in $(seq 1 12); do kill -0 "$P" 2>/dev/null || break; sleep 5; done
    fi
    kill -0 "$P" 2>/dev/null && return 1
  fi
  rm -f serve.pid; return 0
}

# --- start checks (before anything is touched: nothing to restore) ---
N=$(now_min)
if [ -n "${NIGHT_OK:-}" ]; then
  if [ "$NIGHT_OK" = "$(date +%F)" ] && [ "$N" -ge 390 ]; then DEADLINE=$(date -d "tomorrow 06:30" +%s)
  elif [ "$NIGHT_OK" = "$(date -d yesterday +%F)" ] && [ "$N" -lt 390 ]; then DEADLINE=$(date -d "06:30" +%s)
  else log "REFUSE window $WIN: NIGHT_OK=$NIGHT_OK is not tonight"; exit 1; fi
  log "night mode (NIGHT_OK=$NIGHT_OK): deadline $(date -d @"$DEADLINE" '+%F %H:%M'), fan guard ${FAN_LIMIT}% for ${FAN_MAX_S}s"
else
  [ "$N" -ge 480 ] || { log "REFUSE window $WIN: before 08:00"; exit 1; }
  DEADLINE=$(date -d "21:00" +%s)
fi
[ $(( $(date +%s) + EST_MIN * 60 )) -le "$DEADLINE" ] \
  || { log "REFUSE window $WIN: ~${EST_MIN} min from now runs past the $(date -d @"$DEADLINE" +%H:%M) deadline (the closing baseline would be lost)"; exit 1; }
{ [ "$WIN" = D ] || [ "$WIN" = Q ]; } && { [ -d "$STOCK_CKPT" ] && [ -x /mnt/bulk/sigsec-scratch/.venv-1cat151/bin/python ] && [ -x /mnt/bulk/sigsec-scratch/.venv-benchy040/bin/llama-benchy ] \
  || { log "REFUSE window $WIN: stock checkpoint, 1.5.1 env or benchy env missing"; exit 1; }; }
[ "$WIN" = Q ] && { [ -s "$RES/quality-skinny_prod.json" ] || { log "REFUSE window Q: skinny gate (quality-skinny_prod.json) not finished — stopping llama-swap would kill it"; exit 1; }; }
P=$(cat serve.pid 2>/dev/null)
if [ -n "$P" ]; then
  is_vllm "$P" && { log "HARD STOP: a manual :8096 lane is already running (pid $P)"; exit 1; }
  rm -f serve.pid   # stale pidfile from an earlier run; never TERM a reused pid
fi
FOREIGN=$(foreign_on_23)
[ -z "$FOREIGN" ] || { log "HARD STOP: non-llama-swap GPU users on 2,3:$FOREIGN"; exit 1; }
FW_ACTIVE=$(systemctl --user is-active flex-watchdog.timer); FW_PAUSED=0; [ -e /tmp/flex_watchdog.pause ] && FW_PAUSED=1

# --- Telegram warning + 5-min abortable countdown (no send = no window) ---
rm -f "$ABORT"
notify WARN "Stage 0 window $WIN will take GPUs 2,3 (skinny via llama-swap) in 5 min for ~${EST_MIN} min. Abort: touch $ABORT  or  kill -TERM $$" \
  || { log "HARD STOP: warning not delivered"; exit 1; }
for _ in $(seq 30); do [ -e "$ABORT" ] && { log "ABORTED in countdown"; exit 0; }; sleep 10; done
FOREIGN=$(foreign_on_23)
[ -z "$FOREIGN" ] || { log "HARD STOP after countdown: non-llama-swap GPU users on 2,3:$FOREIGN"; exit 1; }

restore() {
  local rc=$? t=0
  trap '' TERM INT   # restore is bounded (~18 min worst case); don't let a second signal leave it half done
  if [ -n "$CH" ] && kill -0 "$CH" 2>/dev/null; then   # a boot/client in flight: native.sh's own trap tears its server down
    kill -TERM "$CH" 2>/dev/null
    for _ in $(seq 1 60); do kill -0 "$CH" 2>/dev/null || break; sleep 2; done
  fi
  [ -n "$SMI" ] && kill "$SMI" 2>/dev/null
  [ -n "$FANW" ] && kill "$FANW" 2>/dev/null
  [ "$STARTED" = 1 ] || return
  stop_server || log "WARN: api_server still alive after bounded stop"
  if [ "$SHED" = 1 ]; then
    log "UPS shed / external restore — not restarting llama-swap; restore-stable owns recovery"
    notify WARN "Stage 0 window $WIN stopped (rc $rc): UPS shed or external restore detected; window did NOT restart llama-swap" || true
    return
  fi
  if ! wait_free 300; then
    log "CRITICAL: GPUs 2,3 still held after 300 s — llama-swap left DOWN"
    notify ERROR "CRITICAL: Stage 0 window $WIN ended (rc $rc) but GPUs 2,3 did not free; llama-swap left DOWN. Check nvidia-smi, then: systemctl --user start llama-swap" || true
    return
  fi
  systemctl --user start llama-swap
  [ "$FW_ACTIVE" = active ] && systemctl --user start flex-watchdog.timer
  [ "$FW_PAUSED" = 0 ] && rm -f /tmp/flex_watchdog.pause
  until curl -s -m 5 http://127.0.0.1:8082/running | grep -q '"model":"qwen3.8-27b-skinny","state":"ready"'; do
    [ "$t" -ge 600 ] && break; sleep 10; t=$((t + 10))
  done
  if [ "$t" -lt 600 ]; then
    log "RESTORED $(date -Is) (skinny ready)"
    notify INFO "Stage 0 window $WIN done (rc $rc); llama-swap restored, skinny ready" || true
  else
    log "WARN: llama-swap started but skinny not ready after 600 s"
    notify WARN "Stage 0 window $WIN done (rc $rc); llama-swap started but skinny NOT ready after 600 s" || true
  fi
}
trap restore EXIT
trap 'log "TERM received${TAG:+ during $TAG}"; exit 143' TERM
trap 'log "INT received${TAG:+ during $TAG}"; exit 130' INT
fan_watch() { # background: chassis fan PWM (iDRAC Oem.Dell.FanPWM) every 30 s -> fans.csv; TERMs the window after
  local hot=0 blind=0 m  #   >= FAN_LIMIT % for FAN_MAX_S, or after FAN_MAX_S with no reading (fail closed)
  while :; do
    m=$(curl -sk -m 15 -K "$HOME/.sigscan/idrac_creds" 'https://192.168.1.2/redfish/v1/Chassis/System.Embedded.1/ThermalSubsystem/Fans?$expand=*($levels=1)' 2>/dev/null \
      | python3 -I -c 'import json,sys; d=json.load(sys.stdin); v=[(x.get("Oem",{}).get("Dell",{}) or {}).get("FanPWM") for x in d.get("Members",[])]; v=[p for p in v if isinstance(p,(int,float))]; print(int(max(v)) if v else "")' 2>/dev/null)
    echo "$(date -Is),${m:-NA}" >> "$RES/fans.csv"
    # leaky counter: a fan hovering 93-97 % still trips (a single dip below the limit only drains 30 s)
    if [[ "$m" =~ ^[0-9]+$ ]]; then blind=0; if [ "$m" -ge "$FAN_LIMIT" ]; then hot=$((hot + 30)); elif [ "$hot" -gt 0 ]; then hot=$((hot - 30)); fi
    else blind=$((blind + 30)); fi
    if [ "$hot" -ge "$FAN_MAX_S" ] || [ "$blind" -ge "$FAN_MAX_S" ]; then
      echo "$(date +%H:%M:%S) FAN GUARD: $([ "$hot" -ge "$FAN_MAX_S" ] && echo "PWM >= ${FAN_LIMIT}% for ${hot}s" || echo "no fan reading for ${blind}s") — stopping window" | tee -a "$LOGF"
      touch "$ABORT"; kill -TERM "$MAIN" 2>/dev/null; return
    fi
    sleep 30
  done
}
STARTED=1
touch /tmp/flex_watchdog.pause; systemctl --user stop flex-watchdog.timer llama-swap
log "WINDOW $WIN START $(date -Is)"
{ [ -n "${NIGHT_OK:-}" ] || [ "$WIN" = D ] || [ "$WIN" = Q ]; } && { fan_watch & FANW=$!; }   # A/B by day unchanged

guard() {
  local row pl dl PLS
  [ -e "$ABORT" ] && { log "ABORT sentinel"; exit 0; }
  [ "$(date +%s)" -lt "$DEADLINE" ] || { log "STOP $(date -d @"$DEADLINE" +%H:%M) deadline guard"; exit 0; }
  [ -z "$FANW" ] || kill -0 "$FANW" 2>/dev/null || { log "STOP: fan watcher died — no fan guard"; exit 1; }
  # sticky shed detection: anything that restarted llama-swap (restore-stable after a UPS shed) owns the cards now
  local ls_state; ls_state=$(systemctl --user is-active llama-swap)
  case "$ls_state" in inactive|failed) ;; *) SHED=1; log "STOP: llama-swap is $ls_state — stack restored under us"; exit 0 ;; esac
  mapfile -t PLS < <(nvidia-smi -i 2,3 --query-gpu=power.limit,power.default_limit --format=csv,noheader,nounits 2>/dev/null)
  [ "${#PLS[@]}" -eq 2 ] || { log "ERROR: power.limit read failed (${#PLS[@]} rows)"; exit 1; }
  for row in "${PLS[@]}"; do
    pl=${row%%,*}; dl=${row#*,}; pl=${pl// /}; dl=${dl// /}; pl=${pl%.*}; dl=${dl%.*}
    [[ "$pl" =~ ^[0-9]+$ && "$dl" =~ ^[0-9]+$ ]] || { log "ERROR: non-numeric power.limit '$row'"; exit 1; }
    [ "$pl" -ge "$dl" ] || { SHED=1; log "STOP power.limit=$pl < default $dl (UPS shed)"; exit 0; }
  done
}
boot() { # boot TAG VAR=val ...
  TAG=$1; shift; guard
  wait_free 300 || { log "ERROR: GPUs 2,3 not free before $TAG"; exit 1; }
  rm -f serve.pid
  nvidia-smi -i 2,3 --query-gpu=timestamp,index,clocks.sm,clocks.mem,power.draw,power.limit,temperature.gpu,clocks_event_reasons.active --format=csv -lms 500 > "$RES/$TAG.smi.csv" & SMI=$!
  run env "$@" LOG="$PWD/$RES/$TAG.serve.log" REASONING_EFFORT=low TP=2 CUDA_VISIBLE_DEVICES=2,3 PORT=8096 MML=204800 GMU=0.95 \
    CUDA_HOME=/usr/local/cuda-12.8 TORCH_EXTENSIONS_DIR=/mnt/bulk/sigsec-scratch/.cache/torch_extensions \
    PATH="$PWD/.venv-sm70/bin:/usr/local/cuda-12.8/bin:$PATH" \
    bash scripts/serve-qwen38-native.sh /mnt/models/RadixArk-Qwen3.8-27B-NVFP4 >> "$RES/$TAG.wrapper.log" 2>&1
}
witness() { # per-arm evidence that the knob moved: CPU/memory placement, custom AR, fusion, AR_EVT
  local w
  {
    echo "## $TAG $(date -Is) env: $*"
    for w in $(nvidia-smi -i 2,3 --query-compute-apps=pid --format=csv,noheader); do
      echo "pid $w $(grep Cpus_allowed_list /proc/$w/status 2>/dev/null)"; numastat -p "$w" 2>/dev/null | tail -1
    done
    grep -m2 -E "Registering [0-9]+ cuda graph addresses|Custom allreduce is disabled" "$RES/$TAG.serve.log"
    grep -m1 -E "fusion replaced [0-9]+ patterns|requires active custom AR" "$RES/$TAG.serve.log"
    grep -m1 -E "\[ar-evt\]" "$RES/$TAG.serve.log"
  } >> "$RES/witness.log" 2>&1
}
alive_or_exit() { is_vllm "$(cat serve.pid 2>/dev/null)" || { log "SERVER DIED during $TAG (external kill or crash)"; exit 1; }; }
stop_arm() {
  stop_server || { log "ERROR: server would not exit after $TAG"; exit 1; }
  [ -n "$SMI" ] && kill "$SMI" 2>/dev/null; SMI=
}
client() { # client TAG — the Stage 0 sweep against whatever arm holds :8096
  run .venv-sm70/bin/python benchmarks/concurrency_sweep.py --model qwen3.8-27b --streams 1,1,1,4 --max-tokens 1024 --tag "s0_$1" \
    >> "$RES/$1.client.log" 2>&1 || log "client rc=$? for $1"
}
benchy() { # benchy TAG — llama-benchy 0.4.0 (pinned venv, local tokenizer, book pre-seeded) at 65K depth, bounded
  run env HF_HUB_OFFLINE=1 timeout -s TERM 1800 /mnt/bulk/sigsec-scratch/.venv-benchy040/bin/llama-benchy \
    --base-url http://127.0.0.1:8096/v1 --model qwen3.8-27b --tokenizer /mnt/models/RadixArk-Qwen3.8-27B-NVFP4 \
    --pp 4096 16384 --tg 256 --depth 65535 --runs 5 --exact-tg \
    --extra-body temperature=0 --extra-body 'chat_template_kwargs={"enable_thinking":false}' \
    --save-result "$RES/$1.benchy.json" --format json >> "$RES/$1.benchy.log" 2>&1 || log "benchy rc=$? for $1"
}
sweep() { # sweep TAG VAR=val ...  (3x 1-stream reps + one 4-stream; + llama-benchy when WITH_BENCHY=1)
  local T=$1
  if boot "$@"; then
    witness "${@:2}"
    client "$T"; alive_or_exit
    [ "$WITH_BENCHY" = 1 ] && { benchy "$T"; alive_or_exit; }
  else log "GATE FAIL $T"; echo "GATE FAIL $T" >> "$RES/failures.txt"; fi
  stop_arm
}
stock() { # stock TAG VAR=val ...  stock 1Cat 1.5.1 lane (scripts/serve-1cat-stock.sh); returns 1 on gate fail
  local T=$1 rc=0; shift; TAG=$T; guard
  wait_free 300 || { log "ERROR: GPUs 2,3 not free before $T"; exit 1; }
  rm -f serve.pid
  nvidia-smi -i 2,3 --query-gpu=timestamp,index,clocks.sm,clocks.mem,power.draw,power.limit,temperature.gpu,clocks_event_reasons.active --format=csv -lms 500 > "$RES/$T.smi.csv" & SMI=$!
  # defaults FIRST, per-arm "$@" LAST: with env, the later assignment wins
  if run env LOG="$PWD/$RES/$T.serve.log" CUDA_VISIBLE_DEVICES=2,3 PORT=8096 TP=2 GMU=0.85 MML=204800 KV=fp8_e4m3 "$@" \
       bash scripts/serve-1cat-stock.sh "$STOCK_CKPT" >> "$RES/$T.wrapper.log" 2>&1; then
    witness "$@"
    if [ "$WIN" = Q ]; then   # same suite skinny ran live on :8082 (quality-skinny_prod.json)
      run timeout -s TERM 3600 /mnt/bulk/sigsec-scratch/.venv-benchy040/bin/python -I benchmarks/quality_gate.py \
        http://127.0.0.1:8096 qwen3.8-27b "$T" >> "$RES/$T.quality.log" 2>&1 || log "quality rc=$? for $T"
      alive_or_exit
    else
      client "$T"; alive_or_exit
      benchy "$T"; alive_or_exit
    fi
  else log "GATE FAIL $T"; echo "GATE FAIL $T" >> "$RES/failures.txt"; rc=1; fi
  stop_arm
  return $rc
}
longctx() { # longctx TAG VAR=val ...  (3 reps at ~48K tokens; prompt_tokens is in each record)
  local T=$1 r
  if boot "$@"; then
    witness "${@:2}"
    for r in 1 2 3; do
      run .venv-sm70/bin/python benchmarks/longctx_probe.py http://127.0.0.1:8096 qwen3.8-27b 36800 >> "$RES/$T.longctx.jsonl" \
        || log "longctx rep $r rc=$? for $T"
    done
    alive_or_exit
  else log "GATE FAIL $T"; echo "GATE FAIL $T" >> "$RES/failures.txt"; fi
  stop_arm
}
B="MNS=4 MBT=8192 VLLM_SM70_E5_PROF=100"
if [ "$WIN" = A ]; then          # K x partition, baseline first and last
  sweep base1_k7_p1024 K=7 DECODE_PARTITION=1024 $B
  for P in 256 1024; do for KK in 1 2 3 4 5; do sweep "k${KK}_p${P}" K=$KK DECODE_PARTITION=$P $B; done; done
  sweep k7_p256 K=7 DECODE_PARTITION=256 $B
  sweep base2_k7_p1024 K=7 DECODE_PARTITION=1024 $B
elif [ "$WIN" = D ]; then        # stock 1Cat 1.5.1 (BF16-LMHead ckpt, fp8_e4m3 KV, GMU .85) vs skinny, self-bracketed
  WITH_BENCHY=1
  sweep base_k7 K=7 DECODE_PARTITION=1024 $B
  if ! stock x151_mtp_k7 ARM=mtp K=7 && grep -q "requires an FP16 LM-head" "$RES/x151_mtp_k7.serve.log" 2>/dev/null; then
    stock x151_mtp_k7_dv0 ARM=mtp K=7 DRAFT_VOCAB_DEFAULT=0   # the public knob 1.2.2 needed on this path
  fi
  stock x151_dflash2_k7 ARM=dflash2 K=7
  sweep base_k7_end K=7 DECODE_PARTITION=1024 $B
elif [ "$WIN" = Q ]; then        # production-switch quality gate: stock 1.5.1 + DFlash2, the config the swap wrapper runs
  stock x151_dflash2_q ARM=dflash2 K=7
else                             # P=512, NUMA arms, AR fusion, AR timing, long context
  for KK in 3 7; do sweep "k${KK}_p512" K=$KK DECODE_PARTITION=512 $B; done
  sweep numa_off_k7_p1024 K=7 DECODE_PARTITION=1024 SKINNY_NUMA=0 $B
  sweep numa1_k7_p1024 K=7 DECODE_PARTITION=1024 NUMA_NODE=1 $B   # memory locality only (vCPUs not host-pinned)
  sweep arfusion_k7_p1024 K=7 DECODE_PARTITION=1024 VLLM_SM70_TP2_AR_GEMMA_RMS_FUSION=1 $B
  sweep arevt_k7_mns1 K=7 DECODE_PARTITION=1024 MNS=1 VLLM_SM70_AR_EVT=1 VLLM_SM70_E5_SUBPROF=1 VLLM_SM70_E5_PROF=100
  for P in 256 1024; do longctx "long_k7_p${P}" K=7 DECODE_PARTITION=$P $B; done
  sweep base3_k7_p1024 K=7 DECODE_PARTITION=1024 $B
fi
TAG=
log "WINDOW $WIN arms complete"
