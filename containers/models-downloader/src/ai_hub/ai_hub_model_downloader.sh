#!/bin/bash

# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0


cd /models

# If the target directory already exists (e.g. a partial download left by a
# SIGKILL, which cannot be intercepted), remove it and start a fresh download.
if [ -d "/models/${model_directory}" ]; then
    echo "{\"event\": \"info\", \"description\": \"Removing existing download and starting fresh: ${model_directory}\"}"
    rm -rf "/models/${model_directory:?}"
fi

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