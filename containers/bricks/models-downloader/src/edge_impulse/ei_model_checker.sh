#!/bin/bash

# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

model_folder="${model_name%.*}"
model_path="/models/${model_folder}"

if [ -f "${model_path}/.download" ]; then
    echo "{\"event\": \"info\", \"description\": \"Model downloading: ${model_name}\", \"downloading\": true}"
    exit 0
elif [ -f "${model_path}/${model_name}" ]; then
    echo "{\"event\": \"info\", \"description\": \"Model exists: ${model_name}\", \"downloading\": false}"
    exit 0
else
    echo "{\"event\": \"error\", \"description\": \"Model does not exist: ${model_name}\", \"downloading\": false}"
    exit 1
fi
