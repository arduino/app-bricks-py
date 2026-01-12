# SPDX-FileCopyrightText: Copyright (C) ARDUINO SRL (http://www.arduino.cc)
#
# SPDX-License-Identifier: MPL-2.0

from .cloud_asr import CloudASR
from .providers import CloudProvider

__all__ = ["CloudASR", "CloudProvider"]
