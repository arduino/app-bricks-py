#!/bin/bash

cd /models

cmd=(python download_ai_hub_model.py
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