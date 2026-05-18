# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

from enum import StrEnum


class CloudModel(StrEnum):
    ANTHROPIC_CLAUDE = "claude-sonnet-4-6"  # https://platform.claude.com/docs/en/about-claude/models/overview#latest-models-comparison
    OPENAI_GPT = "gpt-5.4-mini"  # https://platform.openai.com/docs/models
    GOOGLE_GEMINI = "gemini-2.5-flash"  # https://ai.google.dev/gemini-api/docs/models


class CloudModelProvider(StrEnum):
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    GOOGLE = "google"
