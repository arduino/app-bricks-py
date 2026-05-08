#!/bin/bash

python /app/ei_downloader/download_ei_build.py \
    --json-progress \
    --ei-project-id "${ei_project_id}" \
    --impulse-id "${ei_impulse_id}" \
    --output-name "${model_name}" \
    --output-dir /models \
    --quantization "${quantization}" \
    --target "${target}"
if [ $? -ne 0 ]; then
    echo "Failed to download the model: ${model_name}"
    exit 1
fi
