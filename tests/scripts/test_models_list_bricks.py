# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""Guards between the shipped models list and the bricks it names.

A model served by a local runner says which one in its id, ``genie:`` or ``llamacpp:``.
Every brick the catalog lists such a model under has to declare that framework and
require that runner's service for it, or App Lab offers the model to a brick that
cannot run it.
"""

from __future__ import annotations

from fnmatch import fnmatch
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
BRICKS_DIR = REPO_ROOT / "src" / "arduino" / "app_bricks"
RUNNER_FRAMEWORKS = ("genie", "llamacpp")


def _brick_config(brick_id: str) -> dict:
    return yaml.safe_load((BRICKS_DIR / brick_id.split(":", 1)[1] / "brick_config.yaml").read_text(encoding="utf-8"))


def _catalog() -> dict[str, dict]:
    data = yaml.safe_load((REPO_ROOT / "models" / "models-list.yaml").read_text(encoding="utf-8"))
    return {model_id: model for entry in data["models"] for model_id, model in entry.items()}


def _runner_models_and_bricks() -> list[tuple[str, str]]:
    return sorted(
        (model_id, brick["id"])
        for model_id, model in _catalog().items()
        if model_id.split(":", 1)[0] in RUNNER_FRAMEWORKS
        for brick in model.get("bricks") or []
    )


def _brick_default_models() -> list[tuple[str, str]]:
    """``(brick id, model id)`` for every default model a brick configuration names."""
    defaults = []
    for path in sorted(BRICKS_DIR.glob("*/brick_config.yaml")):
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        models = [config.get("model")] + [board.get("model") for board in config.get("model_by_boards") or []]
        defaults += [(config["id"], model) for model in models if model]
    return defaults


@pytest.mark.parametrize(("brick_id", "model_id"), _brick_default_models())
def test_a_brick_default_model_is_in_the_catalog(brick_id: str, model_id: str):
    assert model_id in _catalog(), f"{brick_id} defaults to {model_id}, which models-list.yaml does not declare"


def _required_services(config: dict, model_id: str) -> set[str]:
    """The services a brick declares for *model_id*: a plain id always, a ``when.model``
    pattern only for the models it matches."""
    services = set()
    for requirement in config.get("requires_services") or []:
        if isinstance(requirement, str):
            services.add(requirement)
        elif fnmatch(model_id, (requirement.get("when") or {}).get("model", "*")):
            services.add(requirement["id"])
    return services


@pytest.mark.parametrize(("model_id", "brick_id"), _runner_models_and_bricks())
def test_a_brick_can_run_every_runner_model_listed_under_it(model_id: str, brick_id: str):
    framework = model_id.split(":", 1)[0]
    config = _brick_config(brick_id)

    assert framework in (config.get("ai_frameworks_compatibility") or []), f"{brick_id} does not declare the {framework} framework"
    assert f"arduino:{framework}" in _required_services(config, model_id), f"{brick_id} does not require arduino:{framework} for {model_id}"
