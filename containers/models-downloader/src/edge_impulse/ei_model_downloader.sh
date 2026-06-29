#!/bin/bash

# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

quantization_arg=()
if [ -n "${quantization}" ]; then
    quantization_arg=(--quantization "${quantization}")
fi

# Each model lives in its own folder named after model_name without its
# extension (e.g. efficientnet-b4-qnn.eim -> efficientnet-b4-qnn). The .eim file
# and the in-progress ".download" marker both live inside this folder, mirroring
# the AI Hub / HF layout.
model_folder="${model_name%.*}"
model_path="/models/${model_folder}"

# A ".download" marker, or a leftover folder that is empty / only contains that
# marker, means a previous run was killed mid-download and must be wiped and
# retried; absent but the file exists => already complete.
if [ -f "${model_path}/.download" ] || { [ -d "${model_path}" ] && [ -z "$(find "${model_path}" -mindepth 1 ! -name '.download' -print -quit 2>/dev/null)" ]; }; then
    echo "{\"event\": \"info\", \"description\": \"Removing incomplete previous download: ${model_folder}\"}"
    rm -rf "${model_path:?}"
elif [ -f "${model_path}/${model_name}" ]; then
    echo "{\"event\": \"info\", \"description\": \"Model exists: ${model_name}\"}"
    exit 0
fi

# Flag this model's download as in-progress; download_ei_build.py removes it on success.
mkdir -p "${model_path}"
printf '%s\n' "${model_folder}" > "${model_path}/.download"

# Use exec so python replaces this shell as PID 1 and receives SIGINT/SIGTERM
# directly, allowing it to clean up partial downloads before exiting.
exec python /app/edge_impulse/download_ei_build.py \
    --ei-project-id "${ei_project_id}" \
    --impulse-id "${ei_impulse_id}" \
    --output-name "${model_name}" \
    --output-dir "${model_path}" \
    "${quantization_arg[@]}" \
    --target "${target}"
