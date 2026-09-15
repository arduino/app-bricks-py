#!/bin/bash

# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

args=(
    --model-url "${model_url}"
    --output-dir /models
    --check
)
if [ -n "${model_mmproj_url}" ]; then
    args+=(--model-mmproj-url "${model_mmproj_url}")
fi

python /app/hugging_face/hf_downloader.py "${args[@]}"
