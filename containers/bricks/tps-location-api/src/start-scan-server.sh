#!/bin/sh

# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

set -e

if [ -z "$PYTHONUNBUFFERED" ]; then
  export PYTHONUNBUFFERED=1
fi

PORT="${SCANNER_PORT:-8089}"

exec python -m uvicorn scan_server:app --host 0.0.0.0 --port "$PORT" --log-level warning
