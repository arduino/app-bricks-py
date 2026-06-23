#!/bin/bash

# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0


# Skip the download only when the model directory already contains every
# file listed in its download manifest (and each file has the expected
# size). A bare or partially-extracted directory is treated as missing
# and silently re-downloaded.
if python /app/ai_hub/ai_hub_model_checker.py > /dev/null 2>&1; then
    echo "{\"event\": \"info\", \"description\": \"Model exists: ${model_directory}\"}"
    exit 0
fi

# Wipe any leftover from a previous incomplete download so the extraction
# starts from a clean slate.
rm -rf "/models/${model_directory}"

cd /models

cmd=(python /app/ai_hub/download_ai_hub_model.py
    --model_type "$model_type"
    --model_name "$model_name"
    --quantization "$quantization"
    --chipset "$chipset"
)
if [ -n "$version" ]; then
    cmd+=(--version "$version")
fi

"${cmd[@]}"
if [ $? -ne 0 ]; then
    echo "{\"event\": \"error\", \"description\": \"Failed to download the model: ${model_name}\"}"
    exit 1
fi