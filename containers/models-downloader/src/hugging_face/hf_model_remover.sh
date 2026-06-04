#!/bin/bash

# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

if [ -n "${model_key}" ]; then
    python /app/hugging_face/hf_downloader.py \
        --model-key "${model_key}" \
        --output-dir /models \
        --delete
    exit_code=$?
    model_id="${model_key}"
else
    args=(
        --model-repo-id "${model_repo_id}"
        --model-name "${model_name}"
        --output-dir /models
        --delete
    )
    if [ -n "${model_mmproj_name}" ]; then
        args+=(--model-mmproj-name "${model_mmproj_name}")
    fi
    python /app/hugging_face/hf_downloader.py "${args[@]}"
    exit_code=$?
    model_id="${model_repo_id}/${model_name}"
fi

if [ "${exit_code}" -ne 0 ]; then
    echo "{\"event\": \"error\", \"description\": \"Failed to remove model: ${model_id}\"}"
    exit 1
fi

echo "{\"event\": \"info\", \"description\": \"Model removed: ${model_id}\"}"
