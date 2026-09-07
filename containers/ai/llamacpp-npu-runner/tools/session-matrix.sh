#!/bin/bash
# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0
#
# Walks every model in the given directories across the context sizes the service runs
# at and, for each, finds the smallest number of Hexagon sessions it loads and generates
# on. Appends one RESULT line per trial to OUTDIR/results.txt and one MATRIX line per
# (model, context) to OUTDIR/matrix.txt.
#
# Resumable: a trial already recorded in results.txt is not run again.
#
# Usage: ./session-matrix.sh [MODELS_DIR ...]
#
# Environment:
#   OUTDIR           where results and logs go (default ./session-matrix)
#   SETTLE           idle seconds before each trial (default 60, see below)
#   CTX_SIZES        context sizes to walk (default "4096 8192 16384")
#   MAX_SESSIONS     largest session count to try (default 4)
#   LLAMACPP_PREFIX  passed through to session-trial.sh
set -u

MODELS_DIRS=("$@")
[ ${#MODELS_DIRS[@]} -eq 0 ] && MODELS_DIRS=(/var/lib/arduino-app-cli/models/llamacpp)

OUTDIR="${OUTDIR:-./session-matrix}"
# The DSP tears its protection domains down asynchronously, so a trial started seconds
# after the previous server was killed can fail to map a buffer that it holds on its own
# once the DSP is idle. 60 seconds was enough for every model measured; 0 reproduces
# what back-to-back loads see, which is also what the router does when it swaps models.
SETTLE="${SETTLE:-60}"
CTX_SIZES="${CTX_SIZES:-4096 8192 16384}"
MAX_SESSIONS="${MAX_SESSIONS:-4}"

TRIAL="$(dirname "$(readlink -f "$0")")/session-trial.sh"
RESULTS="$OUTDIR/results.txt"
MATRIX="$OUTDIR/matrix.txt"
mkdir -p "$OUTDIR"
: >>"$RESULTS"
: >>"$MATRIX"

log() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }

# Every GGUF in the given directories, smallest first so the cheap answers land early,
# skipping the multimodal projectors and any file installed twice.
models() {
    find "${MODELS_DIRS[@]}" -type f -name '*.gguf' ! -name '*mmproj*' -printf '%s\t%p\n' 2>/dev/null |
        sort -n | awk -F'\t' '{n = split($2, part, "/"); if (!seen[part[n]]++) print $2}'
}

for MODEL in $(models); do
    NAME=$(basename "$MODEL")
    GGUF_MB=$(($(stat -c %s "$MODEL") / 1000000))

    # A loaded model costs about 1.6x its GGUF size in RAM, because the DSP buffers come
    # from /dev/dma_heap/system: refuse the ones that would take the board down with them.
    NEED_MB=$((GGUF_MB * 17 / 10))
    FREE_MB=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)
    if [ "$NEED_MB" -gt $((FREE_MB - 512)) ]; then
        log "SKIP $NAME: needs ~${NEED_MB} MB of RAM, only ${FREE_MB} MB available"
        for CTX in $CTX_SIZES; do
            echo "MATRIX model=$NAME gguf_mb=$GGUF_MB ctx=$CTX min_ndev=skipped_ram" >>"$MATRIX"
        done
        continue
    fi

    # The KV cache only grows with the context, so the count found for one context is
    # the floor for the next one up.
    SEED=1
    for CTX in $CTX_SIZES; do
        FOUND=0
        for ((N = SEED; N <= MAX_SESSIONS; N++)); do
            if grep -q "model=$NAME .*ctx=$CTX ndev=$N " "$RESULTS"; then
                LINE=$(grep "model=$NAME .*ctx=$CTX ndev=$N " "$RESULTS" | tail -1)
                log "CACHED $NAME ctx=$CTX ndev=$N"
            else
                log "TRIAL $NAME ctx=$CTX ndev=$N (settling ${SETTLE}s)"
                sleep "$SETTLE"
                LINE=$("$TRIAL" "$MODEL" "$CTX" "$N" "$OUTDIR" 2>>"$OUTDIR/trial-stderr.log")
                echo "$LINE" >>"$RESULTS"
            fi
            log "  -> $(echo "$LINE" | grep -oE 'verdict=[a-z]+ reason=[a-z_]+ quality=[a-z_]+')"
            case "$LINE" in *"verdict=pass"*)
                FOUND=$N
                break
                ;;
            esac
        done
        if [ "$FOUND" -gt 0 ]; then
            SEED=$FOUND
            echo "MATRIX model=$NAME gguf_mb=$GGUF_MB ctx=$CTX min_ndev=$FOUND ${LINE#RESULT }" >>"$MATRIX"
        else
            SEED=$MAX_SESSIONS
            echo "MATRIX model=$NAME gguf_mb=$GGUF_MB ctx=$CTX min_ndev=none_up_to_$MAX_SESSIONS ${LINE#RESULT }" >>"$MATRIX"
        fi
    done
done
log "MATRIX RUN COMPLETE"
