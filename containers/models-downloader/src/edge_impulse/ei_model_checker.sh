#!/bin/bash

# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

if [ -d "/models/${model_name}" ]; then
    echo "{\"event\": \"info\", \"description\": \"Model exists: ${model_name}\"}"
else
    echo "{\"event\": \"error\", \"description\": \"Model does not exist: ${model_name}\"}"
fi
