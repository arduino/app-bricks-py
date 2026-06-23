#!/bin/bash

# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

# Skip the download only when every file produced by a previous run of
# this exact request is still present on disk with the expected size
# (as recorded in the per-request manifest). A partial / corrupted
# previous download fails the check and is silently re-downloaded;
# ``snapshot_download`` then overwrites mismatching files via the HF
# cache.
if /app/hugging_face/hf_model_checker.sh > /dev/null 2>&1; then
    if [ -n "${model_key}" ]; then
        model_id="${model_key}"
    elif [ -n "${model_url}" ]; then
        model_id="${model_url}"
    else
        model_id="${model_repo_id}/${model_name}"
    fi
    echo "{\"event\": \"info\", \"description\": \"Model exists: ${model_id}\"}"
    exit 0
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
