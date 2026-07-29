#!/bin/bash

# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

echo "Generating models.ini..."
python3 /generate_models_ini.py /models

echo "Starting LLama server..."
export LD_LIBRARY_PATH=/opt/pkg-snapdragon/lib
export ADSP_LIBRARY_PATH=/opt/pkg-snapdragon/lib

# Build --device argument from GGML_HEXAGON_NDEV, falling back to the value detected
# from the installed models (default: 1)
if [ -n "${GGML_HEXAGON_NDEV}" ]; then
  NDEV="${GGML_HEXAGON_NDEV}"
  echo "Using externally configured GGML_HEXAGON_NDEV=${NDEV}"
else
  NDEV="$(python3 /generate_models_ini.py /models --print-ndev)"
  NDEV="${NDEV:-1}"
  export GGML_HEXAGON_NDEV="${NDEV}"
  echo "GGML_HEXAGON_NDEV not set: auto-detected ${NDEV} session(s) from installed models"
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

LLAMA_ARGS=(
  --device "$DEVICE_LIST"
  -ngl 100
  --no-mmap
  --models-preset /models/models.ini
)

if [ "${LLAMA_SERVER_SILENT}" = "1" ]; then
  LLAMA_ARGS+=(--log-disable)
fi

exec /opt/pkg-snapdragon/bin/llama-server "${LLAMA_ARGS[@]}"
