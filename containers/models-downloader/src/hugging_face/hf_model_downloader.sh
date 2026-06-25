#!/bin/bash

# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

# Clear any stale/partial download under /models/<repo_id>/ before fetching.
repo_id=""
if [ -n "${model_repo_id}" ]; then
    repo_id="${model_repo_id}"
elif [ -n "${model_key}" ]; then
    repo_id="$(echo "${model_key}" | cut -d: -f2)"
elif [ -n "${model_url}" ]; then
    repo_id="$(echo "${model_url}" | sed -nE 's#https?://huggingface\.co/([^/]+/[^/]+)/(resolve|blob)/.+#\1#p')"
fi

if [ -n "${repo_id}" ] && [ -d "/models/${repo_id}" ]; then
    rm -fr "/models/${repo_id}"
fi

if [ -n "${model_key}" ]; then
    python /app/hugging_face/hf_downloader.py \
        --model-key "${model_key}" \
        --output-dir /models
    exit_code=$?
    model_id="${model_key}"
elif [ -n "${model_url}" ]; then
    args=(
        --model-url "${model_url}"
        --output-dir /models
    )
    if [ -n "${model_mmproj_url}" ]; then
        args+=(--model-mmproj-url "${model_mmproj_url}")
    fi
    python /app/hugging_face/hf_downloader.py "${args[@]}"
    exit_code=$?
    model_id="${model_url}"
else
    args=(
        --model-repo-id "${model_repo_id}"
        --model-name "${model_name}"
        --output-dir /models
    )
    if [ -n "${model_mmproj_name}" ]; then
        args+=(--model-mmproj-name "${model_mmproj_name}")
    fi
    python /app/hugging_face/hf_downloader.py "${args[@]}"
    exit_code=$?
    model_id="${model_repo_id}/${model_name}"
fi

if [ "${exit_code}" -ne 0 ]; then
    echo "{\"event\": \"error\", \"description\": \"Failed to download the model: ${model_id}\"}"
    exit 1
fi
