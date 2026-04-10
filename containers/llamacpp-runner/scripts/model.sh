#!/usr/bin/env bash

# SPDX-FileCopyrightText: Copyright (C) ARDUINO SRL (http://www.arduino.cc)
#
# SPDX-License-Identifier: MPL-2.0

set -e

export XDG_CACHE_HOME=/models

CMD="$1"
SOURCE="$2"
ARG="$3"

usage() {
  echo "Usage:"
  echo "  $0 pull <model> (default docker registry)"
  echo "  $0 pull huggingface <repo/model>"
  echo "  $0 pull docker <repo/model>"
  exit 1
}

if [ -z "$CMD" ]; then
  usage
fi

pull_model() {
  FLAG="$1"
  MODEL="$2"

  echo "Pulling model with llama-pull $FLAG $MODEL"
  LD_LIBRARY_PATH=/usr/local/bin/ /usr/local/bin/llama-pull "$FLAG" "$MODEL"

  # Move model files to /models root
  if [ -d /models/llama.cpp ]; then
    mv /models/llama.cpp/*.gguf /models/ 2>/dev/null || true
    rm -f /models/*.etag
    rm -rf /models/llama.cpp
  fi
}

case "$CMD" in
  pull)
    if [ -z "$SOURCE" ]; then
      usage
    fi

    case "$SOURCE" in
      huggingface)
        [ -z "$ARG" ] && usage
        pull_model "-hf" "$ARG"
        ;;
      docker)
        [ -z "$ARG" ] && usage
        pull_model "-dr" "$ARG"
        ;;
      *)
        # backward compatibility → default docker registry
        pull_model "-dr" "$SOURCE"
        ;;
    esac
    ;;
  *)
    usage
    ;;
esac

echo "Done."
exit 0
