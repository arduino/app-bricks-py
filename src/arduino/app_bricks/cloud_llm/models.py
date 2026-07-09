# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

from enum import StrEnum


class CloudModel(StrEnum):
    ANTHROPIC_CLAUDE = "claude-sonnet-4-6"  # https://platform.claude.com/docs/en/about-claude/models/overview#latest-models-comparison
    OPENAI_GPT = "gpt-5.4-mini"  # https://platform.openai.com/docs/models
    GOOGLE_GEMINI = "gemini-3.5-flash"  # https://ai.google.dev/gemini-api/docs/models


class CloudModelProvider(StrEnum):
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    GOOGLE = "google"


class ReasoningEffort(StrEnum):
    """Discrete reasoning effort levels for reasoning-capable models.

    These map to each provider's native knob:
    - OpenAI: `reasoning_effort`.
    - Gemini 3+: `thinking_level`.
    - Gemini 2.5: mapped to a `thinking_budget` token count (see `EFFORT_TO_BUDGET`).

    For fine-grained control, an explicit integer token budget can be passed
    instead of a level (see `CloudLLM.chat_stream_reasoning`).
    """

    MINIMAL = "minimal"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


# Maps discrete effort levels to a `thinking_budget` token count for Gemini 2.5
# models, which do not support the `thinking_level` parameter. Values stay within
# the budget ranges accepted across gemini-2.5 flash/pro/flash-lite variants.
EFFORT_TO_BUDGET = {
    ReasoningEffort.MINIMAL: 512,
    ReasoningEffort.LOW: 2048,
    ReasoningEffort.MEDIUM: 8192,
    ReasoningEffort.HIGH: 24576,
}
