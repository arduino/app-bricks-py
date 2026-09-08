# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

from dataclasses import dataclass

from arduino.app_bricks.cloud_llm import CloudModel


@dataclass(frozen=True)
class ModelConfig:
    name: CloudModel | str
    provider: str
    requires_api_key: bool = True
    api_key: str | None = None
    base_url: str | None = None
