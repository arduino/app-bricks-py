#!/bin/bash

# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

cd /models

model_folder="${model_name%.*}"

rm -fr "${model_folder}"
if [ $? -ne 0 ]; then
    echo "{\"event\": \"error\", \"description\": \"Failed to remove model: ${model_name}\"}"
    exit 1
fi

echo "{\"event\": \"info\", \"description\": \"Model removed: ${model_name}\"}"
