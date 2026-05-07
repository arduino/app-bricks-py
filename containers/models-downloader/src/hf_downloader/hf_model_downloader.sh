#!/bin/bash

python /app/hf_downloader/hf_downloader.py --model-key ${model_key} --output-dir /models --output-name ${model_name}
if [ $? -ne 0 ]; then
    echo "Failed to download the model: ${model_key}"
    exit 1
fi
