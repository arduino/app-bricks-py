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
  exit 1
}

if [ -z "$CMD" ]; then
  usage
fi

case "$CMD" in
  pull)
    if [ -z "$ARG" ]; then
      usage
    else
      LD_LIBRARY_PATH=/usr/local/bin/ /usr/local/bin/llama-pull -dr "$ARG"

      # Move model files to /models root
      mc /models/llama.cpp/* /models
      rm -f /models/*.etag
      rm -fr /models/llama.cpp
    fi
    ;;
  *)
    usage
    ;;
esac

echo "Done."
exit 0
