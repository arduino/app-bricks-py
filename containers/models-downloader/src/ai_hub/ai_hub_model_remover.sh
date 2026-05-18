#!/bin/bash

# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

cd /models

cmd=(rm -fr "$model_directory")

"${cmd[@]}"
if [ $? -ne 0 ]; then
    echo "Failed to remove model: ${model_directory}"
    exit 1
fi