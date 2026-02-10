#!/usr/bin/env bash

# SPDX-FileCopyrightText: Copyright (C) ARDUINO SRL (http://www.arduino.cc)
#
# SPDX-License-Identifier: MPL-2.0

set -e

CMD="$1"
ARG="$2"

usage() {
  echo "Usage:"
  echo "  $0 pull <model>"
  echo "  $0 rm <model>"
  echo "  $0 ls"
  echo "  $0 ps"
  exit 1
}

if [ -z "$CMD" ]; then
  usage
fi

case "$CMD" in
  pull|rm)
    if [ -z "$ARG" ]; then
      usage
    fi
    ;;
  ls|ps)
    ;;
  *)
    usage
    ;;
esac

ollama serve 2>&1 &
OLLAMA_PID=$!

cleanup() {
  kill "$OLLAMA_PID" 2>/dev/null || true
  wait "$OLLAMA_PID" 2>/dev/null || true
}
trap cleanup EXIT

until ollama list >/dev/null 2>&1; do
  sleep 1
done

ollama "$CMD" ${ARG:+ "$ARG"}

echo "Done."
