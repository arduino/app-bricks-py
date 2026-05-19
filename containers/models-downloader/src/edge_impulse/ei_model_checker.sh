#!/bin/bash

# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

if [ -f "/models/${model_name}" ]; then
    echo "{\"event\": \"info\", \"description\": \"Model exists: ${model_name}\"}"
    exit 0
else
    echo "{\"event\": \"error\", \"description\": \"Model does not exist: ${model_name}\"}"
    exit 1
fi
