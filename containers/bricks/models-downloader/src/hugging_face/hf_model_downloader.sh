#!/bin/bash

# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

# model_url names the model in either syntax: a Hugging Face file URL, or the compact
# "[<model_type>:]<repo_id>:<quantization>[:<mmproj_quantization>]" key. One variable
# covers every case; hf_downloader.py decides which syntax it is.
args=(
    --model-url "${model_url}"
    --output-dir /models
)
if [ -n "${model_mmproj_url}" ]; then
    args+=(--model-mmproj-url "${model_mmproj_url}")
fi

# exec so python becomes PID 1 and receives SIGINT/SIGTERM. hf_downloader.py
# manages the per-repo ".download" marker and wipes partial repos from a kill.
exec python /app/hugging_face/hf_downloader.py "${args[@]}"
