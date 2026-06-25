#!/bin/bash

# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

# Clear any stale/partial download before fetching.
if [ -n "${model_directory}" ] && [ -d "/models/${model_directory}" ]; then
    rm -fr "/models/${model_directory}"
fi

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
