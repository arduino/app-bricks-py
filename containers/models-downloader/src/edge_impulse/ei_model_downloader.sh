#!/bin/bash

# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

quantization_arg=()
if [ -n "${quantization}" ]; then
    quantization_arg=(--quantization "${quantization}")
fi

# Per-model ".${model_name}.download" marker: present => prior run was killed
# mid-download, wipe and retry; absent but file exists => already complete.
if [ -f "/models/.${model_name}.download" ]; then
    echo "{\"event\": \"info\", \"description\": \"Removing incomplete previous download: ${model_name}\"}"
    rm -f "/models/${model_name}"
elif [ -f "/models/${model_name}" ]; then
    echo "{\"event\": \"info\", \"description\": \"Model exists: ${model_name}\"}"
    exit 0
fi

# Flag this model's download as in-progress; download_ei_build.py removes it on success.
printf '%s\n' "${model_name}" > "/models/.${model_name}.download"

# Use exec so python replaces this shell as PID 1 and receives SIGINT/SIGTERM
# directly, allowing it to clean up partial downloads before exiting.
exec python /app/edge_impulse/download_ei_build.py \
    --ei-project-id "${ei_project_id}" \
    --impulse-id "${ei_impulse_id}" \
    --output-name "${model_name}" \
    --output-dir /models \
    "${quantization_arg[@]}" \
    --target "${target}"
