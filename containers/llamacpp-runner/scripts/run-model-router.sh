#!/bin/bash

# SPDX-FileCopyrightText: Copyright (C) ARDUINO SRL (http://www.arduino.cc)
#
# SPDX-License-Identifier: MPL-2.0

echo "Generating models.ini..."
python3 /generate_models_ini.py /models

echo "Starting Llama server..."
export LD_LIBRARY_PATH=/opt/pkg-cpu/lib
export OMP_NUM_THREADS=4
export OMP_WAIT_POLICY=passive
export GOMP_SPINCOUNT=0
exec /opt/pkg-cpu/bin/llama-server \
  --device none \
  -ngl 0 \
  -t 4 \
  -b 512 \
  -ub 128 \
  -c 1024 \
  --log-disable \
  --models-preset /models/models.ini
