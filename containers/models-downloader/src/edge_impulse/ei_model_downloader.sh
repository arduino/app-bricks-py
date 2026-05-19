#!/bin/bash

# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

python /app/edge_impulse/download_ei_build.py \
    --json-progress \
    --ei-project-id "${ei_project_id}" \
    --impulse-id "${ei_impulse_id}" \
    --output-name "${model_name}" \
    --output-dir /models \
    --quantization "${quantization}" \
    --target "${target}"
if [ $? -ne 0 ]; then
    echo "Failed to download the model: ${model_name}"
    exit 1
fi
