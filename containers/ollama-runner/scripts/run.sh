#!/bin/bash

# SPDX-FileCopyrightText: Copyright (C) ARDUINO SRL (http://www.arduino.cc)
#
# SPDX-License-Identifier: MPL-2.0

# Function to handle cleanup on SIGTERM
cleanup() {
    echo "SIGTERM received. Cleaning up..."
    # Kill background processes
    kill "$OLLAMA_PID"
    exit 0
}

# Trap SIGTERM and SIGINT
trap cleanup SIGTERM SIGINT

echo "Starting Ollama serve..."
ollama serve &
OLLAMA_PID=$!

echo "Processes started (Ollama: $OLLAMA_PID). Waiting..."

# Wait for background processes to keep the script running
wait
