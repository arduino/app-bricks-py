#!/bin/bash

# SPDX-FileCopyrightText: Copyright (C) ARDUINO SRL (http://www.arduino.cc)
# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

cd /models

cmd=(python /app/ai_hub_downloader/download_ai_hub_model.py
    --model_type "$model_type"
    --model_name "$model_name"
    --quantization "$quantization"
    --chipset "$chipset"
    --json-progress
)
if [ -n "$version" ]; then
    cmd+=(--version "$version")
fi

"${cmd[@]}"
if [ $? -ne 0 ]; then
    echo "Failed to download the model: ${model_name}"
    exit 1
fi