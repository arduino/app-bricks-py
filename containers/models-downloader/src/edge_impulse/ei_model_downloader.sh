#!/bin/bash

# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

# Skip the download only when the model file is already present and its
# manifest confirms it has the expected size. A bare or partial file is
# treated as missing and silently re-downloaded.
if python /app/edge_impulse/ei_model_checker.py > /dev/null 2>&1; then
    echo "{\"event\": \"info\", \"description\": \"Model exists: ${model_name}\"}"
    exit 0
fi

# Wipe any leftover from a previous incomplete download so the new
# download is not short-circuited by ``http_download.download``'s
# "file already exists" branch.
rm -f "/models/${model_name}" "/models/${model_name}.downloaded.json"

quantization_arg=()
if [ -n "${quantization}" ]; then
    quantization_arg=(--quantization "${quantization}")
fi

python /app/edge_impulse/download_ei_build.py \
    --ei-project-id "${ei_project_id}" \
    --impulse-id "${ei_impulse_id}" \
    --output-name "${model_name}" \
    --output-dir /models \
    "${quantization_arg[@]}" \
    --target "${target}"
if [ $? -ne 0 ]; then
    echo "{\"event\": \"error\", \"description\": \"Failed to download the model: ${model_name}\"}"
    exit 1
fi
