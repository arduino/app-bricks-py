# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

import time
from collections.abc import Callable

import pytest

from arduino.app_internal.pipeline import Pipeline


def _wait_for(predicate: Callable[[], bool], timeout: float = 10.0) -> None:
    deadline = time.time() + timeout
    while not predicate() and time.time() < deadline:
        time.sleep(0.05)


def test_pipeline_moves_items_from_source_to_sink():
    items = iter([1, 2, 3])

    def produce() -> int | None:
        return next(items, None)  # None ends the stream

    def process(x: int) -> int:
        return x * 2

    got: list[int] = []

    def consume(x: int) -> None:
        got.append(x)

    pipeline = Pipeline()
    pipeline.add_source(produce).add_processor(process).add_sink(consume)
    pipeline.start()
    try:
        _wait_for(lambda: len(got) == 3)
    finally:
        pipeline.stop()

    assert got == [2, 4, 6]


def test_pipeline_rejects_bricks_without_the_expected_method():
    class NoProduce:
        pass

    with pytest.raises(TypeError):
        Pipeline().add_source(NoProduce())
