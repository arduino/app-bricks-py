#!/bin/bash

# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

python /app/ai_hub/ai_hub_model_info.py \
    --model-type "${model_type}" \
    --model-name "${model_name}"
