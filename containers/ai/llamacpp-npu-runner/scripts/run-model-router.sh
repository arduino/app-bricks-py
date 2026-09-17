#!/bin/bash

# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

echo "Generating models.ini..."
python3 /configure-llamacpp.py /models

echo "Starting LLama server..."
export LD_LIBRARY_PATH=/opt/pkg-snapdragon/lib
export ADSP_LIBRARY_PATH=/opt/pkg-snapdragon/lib

# The KV cache lives on the DSP domains together with the weights, so a model big enough to
# need several sessions leaves no room for a large context: configure-llamacpp.py caps the
# context for those, exempting the models it knows hold it. A context that is unset, 0 or
# not a number is taken as DEFAULT_CTX_SIZE, the one the service configures out of the box,
# and exported as such: left to itself llama-server would pick a context of its own, and the
# sessions would be sized for a different one than the server runs at.
DEFAULT_CTX_SIZE=16384
REQUESTED_CTX_SIZE="${LLAMA_ARG_CTX_SIZE:-}"
if ! [[ "${REQUESTED_CTX_SIZE}" =~ ^[1-9][0-9]*$ ]]; then
  echo "LLAMA_ARG_CTX_SIZE not configured (was '${REQUESTED_CTX_SIZE}'): using ${DEFAULT_CTX_SIZE}"
  REQUESTED_CTX_SIZE="${DEFAULT_CTX_SIZE}"
  export LLAMA_ARG_CTX_SIZE="${REQUESTED_CTX_SIZE}"
fi

# Ask configure-llamacpp.py one of its two questions (--print-ctx / --print-ndev) for a
# context, and print its answer — or, when it exits badly or answers with anything but a
# number, say so loudly and print the fallback. A crash in there has happened (a segfault
# in its libggml probe), and a command substitution reports none of it: the answer just
# came back empty, which used to pass for "nothing to cap" and "1 session" and started the
# server unable to load the 8B model that needed all four.
size_models() {
  local mode="$1" ctx="$2" fallback="$3" answer status
  answer="$(python3 /configure-llamacpp.py /models "${mode}" --ctx "${ctx}")"
  status=$?
  if [ "${status}" -ne 0 ] || ! [[ "${answer}" =~ ^[0-9]+$ ]]; then
    echo "WARNING: configure-llamacpp.py ${mode} failed (exit status ${status}, answer '${answer}'): falling back to ${fallback}" >&2
    answer="${fallback}"
  fi
  echo "${answer}"
}

EFFECTIVE_CTX_SIZE="$(size_models --print-ctx "${REQUESTED_CTX_SIZE}" "${REQUESTED_CTX_SIZE}")"
if [ "${EFFECTIVE_CTX_SIZE}" != "${REQUESTED_CTX_SIZE}" ]; then
  echo "Big model installed: forcing LLAMA_ARG_CTX_SIZE=${EFFECTIVE_CTX_SIZE} (was ${REQUESTED_CTX_SIZE})"
  export LLAMA_ARG_CTX_SIZE="${EFFECTIVE_CTX_SIZE}"
fi

# Number of Hexagon sessions required by the installed models, sized for the context the
# server will actually run at: more than 1 means at least one model too big for a session.
# When the models cannot be sized at all, every session is configured: too many only
# cost throughput, too few a model that does not load.
MAX_SESSIONS=4
DETECTED_NDEV="$(size_models --print-ndev "${EFFECTIVE_CTX_SIZE}" "${MAX_SESSIONS}")"

# Build --device argument from GGML_HEXAGON_DEVICES (which accepts a session count, like
# the GGML_HEXAGON_NDEV it replaced — still honored for older app configs), falling back
# to the value detected from the installed models (default: 1)
if [ -n "${GGML_HEXAGON_DEVICES}" ]; then
  NDEV="${GGML_HEXAGON_DEVICES}"
  echo "Using externally configured GGML_HEXAGON_DEVICES=${NDEV}"
elif [ -n "${GGML_HEXAGON_NDEV}" ]; then
  NDEV="${GGML_HEXAGON_NDEV}"
  echo "Using externally configured GGML_HEXAGON_NDEV=${NDEV} (deprecated: set GGML_HEXAGON_DEVICES instead)"
else
  NDEV="${DETECTED_NDEV}"
  echo "GGML_HEXAGON_DEVICES not set: auto-detected ${NDEV} session(s) from installed models"
fi
export GGML_HEXAGON_DEVICES="${NDEV}"
# Already translated into GGML_HEXAGON_DEVICES: don't let llama-server see the
# deprecated variable, it would warn on every spawned instance.
unset GGML_HEXAGON_NDEV

echo "Configuring ${NDEV} session(s)..."
DEVICE_LIST=""
for ((i=0; i<NDEV; i++)); do
  if [ -z "$DEVICE_LIST" ]; then
    DEVICE_LIST="HTP${i}"
  else
    DEVICE_LIST="${DEVICE_LIST},HTP${i}"
  fi
done

# NPU offloading can be turned off with LLAMACPP_DISABLE_NPU_SUPPORT=true, which keeps
# every layer on the CPU (-ngl 0). Any other value (default) offloads to the NPU.
if [ "${LLAMACPP_DISABLE_NPU_SUPPORT,,}" = "true" ]; then
  NGL=0
  echo "LLAMACPP_DISABLE_NPU_SUPPORT=true: NPU support disabled, running on CPU (-ngl 0)"
else
  NGL=100
  echo "NPU support enabled (-ngl ${NGL})"
fi

LLAMA_ARGS=(
  --device "$DEVICE_LIST"
  -ngl "$NGL"
  --load-mode none
  --models-preset /models/models.ini
)

if [ "${LLAMA_SERVER_SILENT}" = "1" ]; then
  LLAMA_ARGS+=(--log-disable)
fi

exec /opt/pkg-snapdragon/bin/llama-server "${LLAMA_ARGS[@]}"
