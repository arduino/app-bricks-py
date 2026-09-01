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
# context for those, exempting the models it knows hold it. A context that is unset or not a
# number is left to llama-server, and passed as 0 so that nothing is capped.
REQUESTED_CTX_SIZE="${LLAMA_ARG_CTX_SIZE:-0}"
[[ "${REQUESTED_CTX_SIZE}" =~ ^[0-9]+$ ]] || REQUESTED_CTX_SIZE=0

EFFECTIVE_CTX_SIZE="$(python3 /configure-llamacpp.py /models --print-ctx --ctx "${REQUESTED_CTX_SIZE}")"
EFFECTIVE_CTX_SIZE="${EFFECTIVE_CTX_SIZE:-${REQUESTED_CTX_SIZE}}"
if [ "${EFFECTIVE_CTX_SIZE}" != "${REQUESTED_CTX_SIZE}" ]; then
  echo "Big model installed: forcing LLAMA_ARG_CTX_SIZE=${EFFECTIVE_CTX_SIZE} (was ${REQUESTED_CTX_SIZE})"
  export LLAMA_ARG_CTX_SIZE="${EFFECTIVE_CTX_SIZE}"
fi

# Number of Hexagon sessions required by the installed models, sized for the context the
# server will actually run at: more than 1 means at least one model too big for a session.
DETECTED_NDEV="$(python3 /configure-llamacpp.py /models --print-ndev --ctx "${EFFECTIVE_CTX_SIZE}")"
DETECTED_NDEV="${DETECTED_NDEV:-1}"

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
