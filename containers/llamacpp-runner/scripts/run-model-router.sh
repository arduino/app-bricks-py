#!/bin/bash

# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

echo "Generating models.ini..."
python3 /generate_models_ini.py /models

echo "Starting Llama server..."
export LD_LIBRARY_PATH=/opt/pkg-cpu/lib
exec /opt/pkg-cpu/bin/llama-server \
  --device none \
  -ngl 0 \
  --log-disable \
  --models-preset /models/models.ini
