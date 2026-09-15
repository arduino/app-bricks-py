#!/bin/bash

# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

# Models pinned to an entry of the project's deployment history are looked up on
# the history endpoint, which addresses the build by id alone.
history_arg=()
if [ -n "${history_id}" ]; then
    history_arg=(--history-id "${history_id}")
fi

python /app/edge_impulse/download_ei_build.py \
    --ei-project-id "${ei_project_id}" \
    --impulse-id "${ei_impulse_id}" \
    --output-name "${model_name}" \
    --output-dir /models \
    --quantization "${quantization}" \
    "${history_arg[@]}" \
    --target "${target}" \
    --info
