# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

from enum import StrEnum


class CloudModel(StrEnum):
    ANTHROPIC_CLAUDE = "claude-sonnet-5"  # https://platform.claude.com/docs/en/about-claude/models/overview#latest-models-comparison
    OPENAI_GPT = "gpt-5.6-terra"  # https://platform.openai.com/docs/models
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
    - Anthropic (legacy): mapped to a `thinking` `budget_tokens` count (see `EFFORT_TO_BUDGET`).
    - Anthropic (Opus 4.7+/Sonnet 5+): mapped to `output_config.effort` (see `ANTHROPIC_EFFORT_MAP`).

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


# --- Anthropic extended-thinking tuning --------------------------------------
# Anthropic extended thinking requires a minimum `budget_tokens` of 1024. Used to
# clamp mapped budgets and as the headroom added to `max_tokens` when needed.
ANTHROPIC_MIN_THINKING_BUDGET = 1024

# Budget applied when reasoning is requested for a legacy Anthropic model without an
# explicit effort, since Claude does not think by default and would otherwise stream
# no reasoning.
ANTHROPIC_DEFAULT_THINKING_BUDGET = 2048

# Maps discrete effort levels to Anthropic's `output_config.effort` values, used by the
# adaptive-thinking path on newer models (Opus 4.7+/Sonnet 5+). The levels are shifted up
# one notch relative to Anthropic's own scale: with adaptive thinking, effort `high`
# (Anthropic's default) only "almost always" thinks and skips reasoning on simple prompts,
# so `HIGH` maps to `xhigh` (which always thinks) to ensure the reasoning stream is
# actually populated. Anthropic has no "minimal" effort, so `MINIMAL` folds into `low`.
ANTHROPIC_EFFORT_MAP = {
    ReasoningEffort.MINIMAL: "low",
    ReasoningEffort.LOW: "medium",
    ReasoningEffort.MEDIUM: "high",
    ReasoningEffort.HIGH: "xhigh",
}
