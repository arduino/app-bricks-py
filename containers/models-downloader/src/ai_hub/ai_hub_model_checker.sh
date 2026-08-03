#!/bin/bash

# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

model_path="/models/${model_directory}"

if [ -f "${model_path}/.download" ]; then
    echo "{\"event\": \"info\", \"description\": \"Model downloading: ${model_directory}\", \"downloading\": true}"
    exit 0
# A directory holding only the ".arduino_metadata.yaml" record and no model content
# is a leftover, not an installed model.
elif [ -d "${model_path}" ] && [ -n "$(find "${model_path}" -mindepth 1 ! -name '.arduino_metadata.yaml*' -print -quit 2>/dev/null)" ]; then
    echo "{\"event\": \"info\", \"description\": \"Model exists: ${model_directory}\", \"downloading\": false}"
    exit 0
else
    echo "{\"event\": \"error\", \"description\": \"Model does not exist: ${model_directory}\", \"downloading\": false}"
    exit 1
fi