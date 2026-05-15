#!/bin/bash

# SPDX-FileCopyrightText: Copyright (C) ARDUINO SRL (http://www.arduino.cc)
#
# SPDX-License-Identifier: MPL-2.0

cd /models

rm -fr "${model_name}"
if [ $? -ne 0 ]; then
    echo "Failed to remove model: ${model_name}"
    exit 1
fi
