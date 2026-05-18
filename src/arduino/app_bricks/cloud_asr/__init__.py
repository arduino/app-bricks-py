# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

from .cloud_asr import CloudASR
from .providers import CloudProvider

__all__ = ["CloudASR", "CloudProvider"]
