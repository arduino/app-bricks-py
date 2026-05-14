#!/bin/bash

# SPDX-FileCopyrightText: Copyright (C) ARDUINO SRL (http://www.arduino.cc)
#
# SPDX-License-Identifier: MPL-2.0

echo "Generating models.ini..."
python3 /generate_models_ini.py /models

echo "Starting Llama server..."
export LD_LIBRARY_PATH=/opt/pkg-snapdragon/lib
export ADSP_LIBRARY_PATH=/opt/pkg-snapdragon/lib
exec /opt/pkg-snapdragon/bin/llama-server \
  --device HTP0 \
  -ngl 100 \
  -fa on \
  --no-mmap \
  -t 4 \
  --poll 1000 \
  --ubatch-size 256 \
  --log-disable \
  --models-preset /models/models.ini
