# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""Consistency checks on the shipped `CloudModel` defaults.

The README "Supported Models" table is a machine-consumed contract (agentic coding tools
generate user code from it), so it must carry exactly the enum's raw string IDs. The
shipped defaults must also land on the provider paths the brick is tuned for (adaptive
thinking on Anthropic, `reasoning_effort='none'` with tools on OpenAI, `thinking_level`
on Gemini).
"""

import re
from pathlib import Path

from langchain_google_genai.chat_models import _is_gemini_3_or_later

import arduino.app_bricks.cloud_llm as cloud_llm_package
from arduino.app_bricks.cloud_llm import CloudLLM, CloudModel

README = Path(cloud_llm_package.__file__).parent / "README.md"

# One table row per enum member: | `CloudModel.<NAME>` | `<raw id>` | <docs link> |
_ROW = re.compile(r"^\|\s*`CloudModel\.(\w+)`\s*\|\s*`([^`]+)`\s*\|", re.MULTILINE)


def _readme_supported_models() -> dict[str, str]:
    return dict(_ROW.findall(README.read_text(encoding="utf-8")))


def test_readme_supported_models_table_matches_enum():
    rows = _readme_supported_models()

    assert set(rows) == {m.name for m in CloudModel}, "README table must list every CloudModel member exactly once"
    for name, raw_id in rows.items():
        assert raw_id == CloudModel[name].value, f"README row for CloudModel.{name} drifted from the enum"


def test_default_anthropic_model_takes_the_adaptive_thinking_path():
    assert CloudLLM._anthropic_requires_adaptive(CloudModel.ANTHROPIC_CLAUDE) is True


def test_default_openai_model_turns_reasoning_off_with_tools_on_chat_completions():
    assert CloudLLM._openai_supports_effort_none(CloudModel.OPENAI_GPT) is True


def test_default_gemini_model_takes_the_thinking_level_path():
    assert _is_gemini_3_or_later(CloudModel.GOOGLE_GEMINI) is True
