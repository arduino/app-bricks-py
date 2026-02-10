# SPDX-FileCopyrightText: Copyright (C) ARDUINO SRL (http://www.arduino.cc)
#
# SPDX-License-Identifier: MPL-2.0

from langchain_core.tools import tool
from .local_llm import LocalLLM

__all__ = ["LocalLLM", "tool"]
