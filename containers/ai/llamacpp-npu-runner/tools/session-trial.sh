#!/bin/bash
# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0
#
# One (model, context, sessions) trial: does this model load on this many Hexagon
# sessions at this context size, and does it still generate sense once it has?
#
# Runs llama-server with the settings the service runs it with, so that the answer is
# the one the runner needs (see service_compose.yaml and run-model-router.sh), waits
# for it to come up, and asks two questions. Prints one "RESULT ..." line of key=value
# fields on stdout; exits 0 when the trial passes.
#
# Usage: ./session-trial.sh MODEL.gguf CTX_SIZE SESSIONS [OUTDIR]
#
# Environment:
#   LLAMACPP_PREFIX  where the llama.cpp package is unpacked (default /opt/pkg-snapdragon)
#   PORT             port for the server under test (default 9998)
set -u

# Print this file's header comment as the usage message.
usage() { sed -n '6,$ { /^#/!q; s/^# \?//; p; }' "$0"; }

if [ $# -lt 3 ]; then
    usage
    exit 2
fi

MODEL="$1"
CTX="$2"
SESSIONS="$3"
OUTDIR="${4:-./session-matrix}"
PREFIX="${LLAMACPP_PREFIX:-/opt/pkg-snapdragon}"
PORT="${PORT:-9998}"

for FILE in "$MODEL" "$PREFIX/bin/llama-server"; do
    if [ ! -r "$FILE" ]; then
        echo "session-trial.sh: cannot read $FILE" >&2
        exit 2
    fi
done

TAG="$(basename "$MODEL" .gguf)__c${CTX}__n${SESSIONS}"
RAW="$OUTDIR/$TAG.raw.log"
LOG="$OUTDIR/$TAG.log"
mkdir -p "$OUTDIR"

# --- the environment the service gives the container, from service_compose.yaml
export LD_LIBRARY_PATH="$PREFIX/lib"
export ADSP_LIBRARY_PATH="$PREFIX/lib"
export LLAMA_ARG_HOST=127.0.0.1
export LLAMA_ARG_PORT="$PORT"
export LLAMA_ARG_POLL=1000
export LLAMA_ARG_BATCH=1024
export LLAMA_ARG_UBATCH=256
export LLAMA_ARG_FLASH_ATTN=on
export LLAMA_ARG_CTX_SIZE="$CTX"
export LLAMA_ARG_REASONING=off
export LLAMA_ARG_THINK_BUDGET=128
export LLAMA_ARG_THREADS=4
export LLAMA_ARG_CPU_MASK=0x0f
export GGML_HEXAGON_OPBATCH=2048
export GGML_HEXAGON_MBUF=512
export GGML_HEXAGON_DEVICES="$SESSIONS"
unset GGML_HEXAGON_NDEV LLAMA_ARG_MODELS_DIR LLAMA_LOG_LEVEL LLAMA_ARG_LOG_VERBOSITY

DEVICES="HTP0"
for ((i = 1; i < SESSIONS; i++)); do DEVICES="$DEVICES,HTP$i"; done

# The bracket keeps the pattern from matching the pkill/pgrep command line itself.
SERVER_PATTERN="$PREFIX/bin/llama-serve[r]"

kill_server() {
    pkill -f "$SERVER_PATTERN" 2>/dev/null
    for _ in $(seq 1 40); do
        pgrep -f "$SERVER_PATTERN" >/dev/null || return 0
        sleep 1
    done
    pkill -9 -f "$SERVER_PATTERN" 2>/dev/null
    sleep 3
}

kill_server
FREE_BEFORE=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)

# -v: the projected per-session memory breakdown is only logged at full verbosity, and
# it is logged before anything is allocated, so it is there for the loads that fail too.
"$PREFIX/bin/llama-server" -m "$MODEL" --device "$DEVICES" -ngl 100 --load-mode none -v \
    >"$RAW" 2>&1 </dev/null &
SERVER=$!

STATE=timeout
for i in $(seq 1 450); do
    if ! kill -0 $SERVER 2>/dev/null; then STATE=died; break; fi
    if curl -s -m 2 "localhost:$PORT/health" 2>/dev/null | grep -q '"ok"'; then STATE=up; break; fi
    sleep 2
done
LOAD_S=$((i * 2))

ANSWER=""
CAPITAL=""
if [ "$STATE" = up ]; then
    ask() {
        curl -s -m 900 "localhost:$PORT/v1/chat/completions" -H "Content-Type: application/json" \
            -d "{\"messages\":[{\"role\":\"user\",\"content\":\"$1\"}],\"max_tokens\":$2,\"temperature\":0,\"stream\":false}"
    }
    content() {
        python3 -c 'import json,sys
try:
    print(json.load(open(sys.argv[1]))["choices"][0]["message"]["content"].strip().replace("\n"," ")[:4000])
except Exception as e:
    print("PARSE_ERROR %s" % e)' "$1"
    }
    ask "Tell me the steps to prepare and cook an apple pie. Use a max of 150 words." 400 >"$OUTDIR/$TAG.pie.json"
    ANSWER=$(content "$OUTDIR/$TAG.pie.json")
    ask "What is the capital of France? Answer with the city name only." 256 >"$OUTDIR/$TAG.capital.json"
    CAPITAL=$(content "$OUTDIR/$TAG.capital.json")
fi
FREE_AFTER=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)

# --- verdict: whether the model loaded on SESSIONS sessions and generated at all.
# Answer quality is reported separately, so that a reasoning model rambling past its
# token budget is not mistaken for an allocation failure.
VERDICT=fail
REASON=""
QUALITY=na
case "$STATE" in
died) REASON=server_exited ;;
timeout) REASON=load_timeout ;;
up)
    if echo "$ANSWER" | grep -q PARSE_ERROR; then
        REASON=no_completion
    elif [ ${#ANSWER} -lt 20 ]; then
        REASON=empty_answer
    else
        VERDICT=pass
        REASON=ok
    fi
    if ! echo "$CAPITAL" | grep -qi paris; then
        QUALITY=wrong_capital
    elif ! echo "$ANSWER" | grep -qiE 'apple|pie|oven|flour|crust|dough'; then
        QUALITY=off_topic
    elif [ ${#ANSWER} -lt 200 ]; then
        QUALITY=short_answer
    else
        QUALITY=ok
    fi
    ;;
esac

# --- keep the interesting lines, drop the rest of the verbose log
grep -aE 'buffer size|memory breakdown|HTP[0-9]+ \(Hexagon\)|CPU_REPACK|llama_kv_cache:|llama_memory_recurrent:|op batching|new session|hwinfo|print_timing|load_model:|params_fit_impl|fastrpc|Invoke Failed|failed|error|print_info: (file size|n_layer|n_embd|n_head|arch|model type|model params)' \
    "$RAW" >"$LOG" 2>/dev/null
rm -f "$RAW"

# --- the numbers worth a matrix, parsed out of the compacted log
eval "$(
    python3 - "$LOG" <<'PY'
import re, shlex, sys

text = open(sys.argv[1], errors="replace").read()


def last(pattern, default=""):
    found = re.findall(pattern, text)
    return found[-1] if found else default


# llama.cpp logs a dry-run pass before the real one: keep the last breakdown of each device
rows = re.findall(r"- (HTP\d+) \(Hexagon\)\s*\|\s*\d+ =\s*\d+ \+ \(\s*(\d+) =\s*(\d+) \+\s*(\d+) \+\s*(\d+)\)", text)
per_device = {}
for device, total, weights, context, compute in rows:
    per_device[device] = (int(total), int(weights), int(context), int(compute))
totals = [sum(values[i] for values in per_device.values()) for i in range(4)]

fields = {
    "htp": ",".join("%s:%d=%d+%d+%d" % ((device,) + per_device[device]) for device in sorted(per_device)) or "-",
    "htp_total_mib": totals[0],
    "htp_model_mib": totals[1],
    "htp_ctx_mib": totals[2],
    "htp_compute_mib": totals[3],
    "kv_mib": last(r"llama_kv_cache: size =\s*([\d.]+) MiB") or "?",
    "kv_cells": last(r"llama_kv_cache: size =.*?\(\s*(\d+) cells") or "?",
    "kv_layers": last(r"llama_kv_cache: size =.*?(\d+) layers") or "?",
    "rs_mib": last(r"llama_memory_recurrent: size =\s*([\d.]+) MiB") or "0",
    "cpu_repack_mib": last(r"CPU_REPACK\s*\|\s*(\d+)") or "0",
    "vmem": last(r"op batching:.*vmem (\d+)") or "?",
    "n_slots": last(r"n_slots = (\d+)") or "?",
    "n_ctx_slot": last(r"n_ctx_slot = (\d+)") or "?",
    "pp_tps": last(r"prompt eval time =.*?,\s*([\d.]+) tokens per second") or "?",
    "tg_tps": last(r"\|\s+eval time =.*?,\s*([\d.]+) tokens per second") or "?",
}
print("\n".join("F_%s=%s" % (key, shlex.quote(str(value))) for key, value in fields.items()))
PY
)"

ERRORS=$(grep -aoE 'fastrpc_mmap failed|failed to allocate|alloc_tensor_range|unable to allocate|Invoke Failed|std::bad_alloc|terminate called' "$LOG" |
    sort | uniq -c | tr '\n' ';' | tr -s ' ')

kill_server
sleep 2

printf 'RESULT model=%s gguf_mb=%s ctx=%s ndev=%s verdict=%s reason=%s quality=%s state=%s load_s=%s n_slots=%s n_ctx_slot=%s htp_total_mib=%s htp_model_mib=%s htp_ctx_mib=%s htp_compute_mib=%s kv_mib=%s kv_cells=%s kv_layers=%s rs_mib=%s cpu_repack_mib=%s vmem=%s pp_tps=%s tg_tps=%s per_session=[%s] free_before_mb=%s free_after_mb=%s errs=[%s] capital=[%s]\n' \
    "$(basename "$MODEL")" "$(($(stat -c %s "$MODEL") / 1000000))" "$CTX" "$SESSIONS" \
    "$VERDICT" "$REASON" "$QUALITY" "$STATE" "$LOAD_S" \
    "$F_n_slots" "$F_n_ctx_slot" "$F_htp_total_mib" "$F_htp_model_mib" "$F_htp_ctx_mib" "$F_htp_compute_mib" \
    "$F_kv_mib" "$F_kv_cells" "$F_kv_layers" "$F_rs_mib" "$F_cpu_repack_mib" "$F_vmem" "$F_pp_tps" "$F_tg_tps" \
    "$F_htp" "$FREE_BEFORE" "$FREE_AFTER" "$ERRORS" "$CAPITAL"
printf '%s\n' "$ANSWER" >"$OUTDIR/$TAG.answer.txt"

[ "$VERDICT" = pass ]
