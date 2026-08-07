#!/bin/bash

# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

echo "Generating models.ini..."
python3 /generate_models_ini.py /models

echo "Starting LLama server..."
export LD_LIBRARY_PATH=/opt/pkg-snapdragon/lib
export ADSP_LIBRARY_PATH=/opt/pkg-snapdragon/lib

# Number of Hexagon sessions required by the installed models, estimated from the size of
# their GGUF files: more than 1 means at least one model too big for a single session.
DETECTED_NDEV="$(python3 /generate_models_ini.py /models --print-ndev)"
DETECTED_NDEV="${DETECTED_NDEV:-1}"

# Build --device argument from GGML_HEXAGON_NDEV, falling back to the value detected
# from the installed models (default: 1)
if [ -n "${GGML_HEXAGON_NDEV}" ]; then
  NDEV="${GGML_HEXAGON_NDEV}"
  echo "Using externally configured GGML_HEXAGON_NDEV=${NDEV}"
else
  NDEV="${DETECTED_NDEV}"
  export GGML_HEXAGON_NDEV="${NDEV}"
  echo "GGML_HEXAGON_NDEV not set: auto-detected ${NDEV} session(s) from installed models"
fi

# Big models leave little room for the KV cache on the NPU: cap their context size. Three or
# more sessions means a GGUF larger than 3.5 GB, which is where the cap starts to be needed.
BIG_MODEL_MIN_NDEV=3
BIG_MODEL_MAX_CTX_SIZE=16384
if [ "${DETECTED_NDEV}" -ge "${BIG_MODEL_MIN_NDEV}" ] && [[ "${LLAMA_ARG_CTX_SIZE}" =~ ^[0-9]+$ ]] && [ "${LLAMA_ARG_CTX_SIZE}" -gt "${BIG_MODEL_MAX_CTX_SIZE}" ]; then
  echo "Big model installed (${DETECTED_NDEV} sessions): forcing LLAMA_ARG_CTX_SIZE=${BIG_MODEL_MAX_CTX_SIZE} (was ${LLAMA_ARG_CTX_SIZE})"
  export LLAMA_ARG_CTX_SIZE="${BIG_MODEL_MAX_CTX_SIZE}"
fi

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
