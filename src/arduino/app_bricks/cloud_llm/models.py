# SPDX-FileCopyrightText: Copyright (C) ARDUINO SRL (http://www.arduino.cc)
#
# SPDX-License-Identifier: MPL-2.0

from enum import Enum


class CloudModel(str, Enum):
    ANTHROPIC_CLAUDE = "claude-3-7-sonnet-latest"
    OPENAI_GPT = "gpt-4o-mini"
    GOOGLE_GEMINI = "gemini-2.5-flash"
