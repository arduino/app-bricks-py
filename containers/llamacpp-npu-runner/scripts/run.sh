#!/bin/bash

# SPDX-FileCopyrightText: Copyright (C) ARDUINO SRL (http://www.arduino.cc)
#
# SPDX-License-Identifier: MPL-2.0

# Function to handle cleanup on SIGTERM
cleanup() {
    echo "SIGTERM received. Cleaning up..."
    # Kill background processes
    kill "$LLAMA_PID"
    exit 0
}

# Trap SIGTERM and SIGINT
trap cleanup SIGTERM SIGINT

echo "Starting Llama server..."
# Add your specific flags here (model path, port, etc.)
export LD_LIBRARY_PATH=/opt/pkg-snapdragon/lib
export ADSP_LIBRARY_PATH=/opt/pkg-snapdragon/lib
# --reasoning off
/opt/pkg-snapdragon/bin/llama-server --device HTP0 --log-disable &
LLAMA_PID=$!

echo "Processes started (Llama: $LLAMA_PID). Waiting..."

# Wait for background processes to keep the script running
wait
