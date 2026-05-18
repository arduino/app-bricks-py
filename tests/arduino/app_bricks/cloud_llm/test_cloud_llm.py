# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

import pytest
from typing import cast

from conftest import load_dataset

from deepeval import assert_test
from deepeval.test_case import LLMTestCaseParams, ToolCallParams
from deepeval.metrics import BaseMetric, GEval, ToolCorrectnessMetric


dataset = load_dataset()


@pytest.mark.parametrize(
    "test_case",
    dataset.test_cases,
)
def test_cloud_llm(test_case, execute_runner, judge_model):
    meta = cast(dict, test_case.additional_metadata)

    response_text, available_tools, tools_called = execute_runner(
        name=meta["runner_name"],
        model=meta["model"],
        prompt=test_case.input,
    )

    test_case.actual_output = response_text
    test_case.tools_called = tools_called

    metrics: list[BaseMetric] = []

    if test_case.expected_tools:
        metrics.append(
            ToolCorrectnessMetric(
                strict_mode=True,
                available_tools=available_tools,
                model=judge_model,
                evaluation_params=[ToolCallParams.INPUT_PARAMETERS],
                verbose_mode=True,
            )
        )

    semantic_test = meta.get("semantic", None)
    if semantic_test:
        criteria = semantic_test.get("criteria", None)
        evaluation_steps = semantic_test.get("evaluation_steps", [])
        min_score = semantic_test.get("min_score", 0.75)
        metrics.append(
            GEval(
                name="semantic evaluation",
                model=judge_model,
                evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
                criteria=criteria,
                evaluation_steps=evaluation_steps,
                threshold=min_score,
                verbose_mode=True,
            )
        )

    assert_test(test_case, metrics)
