#!/bin/bash

# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

NODE_COMMAND=("node" "/app/linux/node/build/cli/linux/runner.js" "$@")

# Bash runs a trap only once the foreground command returns, so the runner is
# started in the background and the signal is forwarded to it: the container
# stops when node exits instead of waiting for the kill timeout.
trap_signal() {
  echo "Caught signal $1. Stopping the runner..."
  kill -s "$1" "$NODE_PID" 2>/dev/null
  wait "$NODE_PID"
  exit 0
}

trap 'trap_signal TERM' TERM
trap 'trap_signal INT' INT

while true; do
  echo "🚀 Starting EI inference runner..."

  "${NODE_COMMAND[@]}" &
  NODE_PID=$!
  wait "$NODE_PID"
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