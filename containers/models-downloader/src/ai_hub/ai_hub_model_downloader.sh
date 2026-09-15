#!/bin/bash

# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0


cd /models

model_path="/models/${model_directory}"

# Decide whether a usable model is already present. A ".download" marker, or a
# leftover directory holding no model content (only the marker and/or the
# ".arduino_metadata.yaml" record, including the ".tmp" sibling of an interrupted
# atomic write), means a previous run was interrupted (e.g. SIGKILL) and must be
# wiped and retried rather than reported as "Model exists".
if [ -f "${model_path}/.download" ] || { [ -d "${model_path}" ] && [ -z "$(find "${model_path}" -mindepth 1 ! -name '.download' ! -name '.arduino_metadata.yaml*' -print -quit 2>/dev/null)" ]; }; then
    echo "{\"event\": \"info\", \"description\": \"Removing incomplete previous download: ${model_directory}\"}"
    rm -rf "${model_path:?}"
elif [ -d "${model_path}" ]; then
    echo "{\"event\": \"info\", \"description\": \"Model exists: ${model_directory}\"}"
    exit 0
fi

# Ensure the model directory exists
mkdir -p "${model_path}"

cmd=(python /app/ai_hub/download_ai_hub_model.py
    --model_type "$model_type"
    --model_name "$model_name"
    --quantization "$quantization"
    --chipset "$chipset"
)
if [ -n "$version" ]; then
    cmd+=(--version "$version")
fi

# Use exec so python replaces this shell as PID 1 and receives SIGINT/SIGTERM
# directly, allowing it to clean up partial downloads before exiting.
exec "${cmd[@]}"