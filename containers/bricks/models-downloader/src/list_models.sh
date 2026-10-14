#!/bin/bash

# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

# --write also materializes the listing as /models/models.yaml (see list_models.py)
python /app/list_models.py --json --supported-board ${BOARD_NAME} --write
