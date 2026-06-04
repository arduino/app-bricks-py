#!/bin/bash

# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

echo "Generating models.ini..."
python3 /generate_models_ini.py /models

echo "Starting Llama server..."
export LD_LIBRARY_PATH=/opt/pkg-snapdragon/lib
export ADSP_LIBRARY_PATH=/opt/pkg-snapdragon/lib

LLAMA_ARGS=(
  --device HTP0
  -ngl 100
  --no-mmap
  --models-preset /models/models.ini
)

if [ "${LLAMA_SERVER_SILENT}" = "1" ]; then
  LLAMA_ARGS+=(--log-disable)
fi

exec /opt/pkg-snapdragon/bin/llama-server "${LLAMA_ARGS[@]}"
