#!/bin/sh

# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

set -eu

SOCKET_PATH="${SCANNER_SOCKET_PATH:-/app/.cache/tps_location_api/scanner.sock}"
SOCKET_DIR="$(dirname "$SOCKET_PATH")"

# The socket directory is shared with the app container only: keep it private to the service user
mkdir -p "$SOCKET_DIR"
chmod 700 "$SOCKET_DIR" || echo "warning: cannot restrict permissions of $SOCKET_DIR" >&2
rm -f "$SOCKET_PATH"

exec python -m uvicorn scan_server:app --uds "$SOCKET_PATH" --log-level warning
