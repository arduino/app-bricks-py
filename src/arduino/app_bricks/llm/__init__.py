# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

from langchain_core.tools import tool
from .local_llm import LargeLanguageModel

__all__ = ["LargeLanguageModel", "tool"]
