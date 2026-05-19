#!/bin/bash

# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

if [ -d "/models/${model_directory}" ]; then
    echo "{\"event\": \"info\", \"description\": \"Model exists: ${model_directory}\"}"
    exit 0
else
    echo "{\"event\": \"error\", \"description\": \"Model does not exist: ${model_directory}\"}"
    exit 1
fi