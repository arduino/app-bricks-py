#!/bin/bash

python /app/ei_downloader/download_ei_build.py --json-progress --ei-project-id ${ei-project-id} --impulse-id ${ei-impulse-id} --output-name ${model_name} --output-dir /models --output-name ${model_name} --quantization ${quantization} --target ${target}
if [ $? -ne 0 ]; then
    echo "Failed to download the model: ${model_name}"
    exit 1
fi
