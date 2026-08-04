#!/bin/bash

# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

args=(
    --model-url "${model_url}"
    --output-dir /models
    --delete
)
if [ -n "${model_mmproj_url}" ]; then
    args+=(--model-mmproj-url "${model_mmproj_url}")
fi

python /app/hugging_face/hf_downloader.py "${args[@]}"
exit_code=$?

if [ "${exit_code}" -ne 0 ]; then
    echo "{\"event\": \"error\", \"description\": \"Failed to remove model: ${model_url}\"}"
    exit 1
fi

echo "{\"event\": \"info\", \"description\": \"Model removed: ${model_url}\"}"
