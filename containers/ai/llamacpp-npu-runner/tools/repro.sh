#!/bin/bash
# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0
#
# Repeats one or more (model, context, sessions) trials to see whether their verdict is
# a property of the model or of the moment. Near the allocation envelope it is not
# always the former: see SESSION_ALLOCATION.md.
#
# The repetitions are interleaved rather than run back to back, so that each one has a
# different predecessor, and every trial gets the same idle time before it.
#
# Usage: ./repro.sh REPS 'MODEL.gguf CTX SESSIONS' ['MODEL.gguf CTX SESSIONS' ...]
#
# Example:
#   ./repro.sh 3 "$MODELS/granite-4.2-3b-Q4_0.gguf 8192 1" \
#                "$MODELS/Qwen_Qwen3-8B-Q4_0.gguf 16384 4"
#
# Environment:
#   OUTDIR           where results and logs go (default ./session-repro)
#   SETTLE           idle seconds before each trial (default 60)
#   LLAMACPP_PREFIX  passed through to session-trial.sh
set -u

# Print this file's header comment as the usage message.
usage() { sed -n '6,$ { /^#/!q; s/^# \?//; p; }' "$0"; }

if [ $# -lt 2 ]; then
    usage
    exit 2
fi

REPS="$1"
shift
CASES=("$@")

OUTDIR="${OUTDIR:-./session-repro}"
SETTLE="${SETTLE:-60}"
TRIAL="$(dirname "$(readlink -f "$0")")/session-trial.sh"
RESULTS="$OUTDIR/results.txt"
mkdir -p "$OUTDIR"
: >>"$RESULTS"

for ((rep = 1; rep <= REPS; rep++)); do
    for CASE in "${CASES[@]}"; do
        # shellcheck disable=SC2086 # the case is three space-separated fields
        set -- $CASE
        printf '[%s] rep %s: %s ctx=%s ndev=%s (settling %ss)\n' \
            "$(date +%H:%M:%S)" "$rep" "$(basename "$1")" "$2" "$3" "$SETTLE"
        sleep "$SETTLE"
        LINE=$("$TRIAL" "$1" "$2" "$3" "$OUTDIR/rep$rep" 2>>"$OUTDIR/trial-stderr.log")
        echo "rep=$rep $LINE" >>"$RESULTS"
        echo "  -> $(echo "$LINE" | grep -oE 'verdict=[a-z]+ reason=[a-z_]+')"
    done
done
echo "REPRO RUN COMPLETE"
