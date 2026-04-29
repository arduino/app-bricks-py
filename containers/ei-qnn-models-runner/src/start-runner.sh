#!/bin/bash

# SPDX-FileCopyrightText: Copyright (C) ARDUINO SRL (http://www.arduino.cc)
#
# SPDX-License-Identifier: MPL-2.0

NODE_COMMAND=("node" "/app/linux/node/build/cli/linux/runner.js" "$@")

trap_signal() {
  echo "Caught signal $1. Exiting wrapper immediately..."
  exit 0
}

trap 'trap_signal TERM' TERM
trap 'trap_signal INT' INT

while true; do
  echo "🚀 Starting EI inference runner..."

  "${NODE_COMMAND[@]}"

  EXIT_CODE=$?

  # Check the exit code
  if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Application exited successfully (Exit Code: 0). Stopping restart loop."
    break
  else
    echo "⚠️ Application exited with error (Exit Code: $EXIT_CODE). Restarting in 1 seconds..."
    sleep 1
  fi
done