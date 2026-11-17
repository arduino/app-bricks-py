#!/bin/bash

# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

args=(
    --model-url "${model_url}"
    --output-dir /models
    --info
)
if [ -n "${model_mmproj_url}" ]; then
    args+=(--model-mmproj-url "${model_mmproj_url}")
fi
# hf_token opens the private and gated repositories it can read; without it the Hub is
# asked anonymously and only public repositories can be sized.
if [ -n "${hf_token}" ]; then
    args+=(--hf-token "${hf_token}")
fi

python /app/hugging_face/hf_downloader.py "${args[@]}"
