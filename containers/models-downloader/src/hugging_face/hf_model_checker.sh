#!/bin/bash

# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

if [ -n "${model_key}" ]; then
    python /app/hugging_face/hf_downloader.py \
        --model-key "${model_key}" \
        --output-dir /models \
        --check
else
    args=(
        --model-repo-id "${model_repo_id}"
        --model-name "${model_name}"
        --output-dir /models
        --check
    )
    if [ -n "${model_mmproj_name}" ]; then
        args+=(--model-mmproj-name "${model_mmproj_name}")
    fi
    python /app/hugging_face/hf_downloader.py "${args[@]}"
fi
