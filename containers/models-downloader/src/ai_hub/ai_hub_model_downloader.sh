#!/bin/bash

# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0


cd /models

# Per-model ".download" marker (inside the model dir): present => prior run was
# killed mid-download, wipe and retry; absent but dir exists => already complete.
if [ -f "/models/${model_directory}/.download" ]; then
    echo "{\"event\": \"info\", \"description\": \"Removing incomplete previous download: ${model_directory}\"}"
    rm -rf "/models/${model_directory:?}"
elif [ -d "/models/${model_directory}" ]; then
    echo "{\"event\": \"info\", \"description\": \"Model exists: ${model_directory}\"}"
    exit 0
fi

# Flag this model's download as in-progress; download_ai_hub_model.py removes it on success.
mkdir -p "/models/${model_directory}"
printf '%s\n' "${model_directory}" > "/models/${model_directory}/.download"

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