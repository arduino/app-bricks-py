#!/bin/bash

# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

cd /models

rm -fr "${model_name}"
if [ $? -ne 0 ]; then
    echo "{\"event\": \"error\", \"description\": \"Failed to remove model: ${model_name}\"}"
    exit 1
fi

echo "{\"event\": \"info\", \"description\": \"Model removed: ${model_name}\"}"
