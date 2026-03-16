#!/bin/sh

# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

REQUIREMENTS_FILE="requirements.txt"

if [ -f "$REQUIREMENTS_FILE" ]; then
    python -m pip install -r "$REQUIREMENTS_FILE"
fi

exec python main.py --verbose --input websocket --output mjpeg websocket