# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""Guards the CPU thread setting of the llama.cpp runner on the UNO Q and the way an app changes it.

Measured on quattro (UNO Q, 2026-11-20), tokens/s in generation while a synthetic process keeps a share of
one core busy. Four threads are the fastest on free cores (8.7 against 6.8 with three) and degrade as soon
as one core is shared (5.0 at 50%, 1.3 at 100%), where three threads stay flat at 6.9. Putting idle threads
to sleep at once (OMP_WAIT_POLICY=passive, GOMP_SPINCOUNT=0) loses on free cores (7.8) and loses to three
threads under load.
"""

import re
import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[4]
# Make the repo-root ``docs_generator`` package importable regardless of the cwd.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from docs_generator.extractor import extract_docstrings_with_types  # noqa: E402

BRICKS_DIR = REPO_ROOT / "src" / "arduino" / "app_bricks"
SERVICE_DIR = REPO_ROOT / "src" / "arduino" / "app_services" / "llamacpp"
UNOQ_COMPOSE = SERVICE_DIR / "service_compose.unoq.yaml"
COMPOSE_FILES = [UNOQ_COMPOSE, SERVICE_DIR / "service_compose.yaml"]
THREADS = "LLAMA_ARG_THREADS"


def _brick_config(brick: str) -> dict:
    return yaml.safe_load((BRICKS_DIR / brick / "brick_config.yaml").read_text(encoding="utf-8"))


def _unoq_llamacpp_bricks() -> list[str]:
    bricks = []
    for config_file in sorted(BRICKS_DIR.glob("*/brick_config.yaml")):
        config = yaml.safe_load(config_file.read_text(encoding="utf-8"))
        if "unoq" in config.get("supported_boards", []) and "llamacpp" in config.get("ai_frameworks_compatibility", []):
            bricks.append(config_file.parent.name)
    return bricks


def _environment(compose_file: Path) -> dict[str, str]:
    (service,) = yaml.safe_load(compose_file.read_text(encoding="utf-8"))["services"].values()
    return dict(entry.split("=", 1) for entry in service["environment"])


def _default(value: str | None) -> str | None:
    """Returns the value a compose entry takes when nothing overrides its variable."""
    match = re.fullmatch(r"\$\{\w+:-(.*)\}", value or "")
    return match.group(1) if match else value


def _compose_default_threads(compose_file: Path) -> str | None:
    value = _environment(compose_file)[THREADS]
    assert value.startswith("${" + THREADS + ":-"), f"{compose_file.name} must let the app override {THREADS}"
    return _default(value)


def test_unoq_openmp_pool_follows_the_thread_setting():
    environment = _environment(UNOQ_COMPOSE)
    assert environment["OMP_NUM_THREADS"] == environment[THREADS], "an app that lowers the threads must shrink the OpenMP pool too"


def test_unoq_idle_threads_are_not_put_to_sleep_at_once():
    environment = _environment(UNOQ_COMPOSE)
    assert _default(environment.get("OMP_WAIT_POLICY")) != "passive"
    assert _default(environment.get("GOMP_SPINCOUNT")) != "0"


def test_there_are_unoq_llamacpp_bricks():
    assert {"llm", "vlm", "decision_model"} <= set(_unoq_llamacpp_bricks())


@pytest.mark.parametrize("brick", _unoq_llamacpp_bricks())
def test_brick_declares_the_threads_with_the_runner_default(brick: str):
    variables = {variable["name"]: variable for variable in _brick_config(brick).get("variables", [])}
    assert THREADS in variables, f"App Lab rejects a variable that {brick} does not declare"
    # The brick default reaches the runner of every board the brick supports
    for compose_file in COMPOSE_FILES:
        assert str(variables[THREADS]["default_value"]) == _compose_default_threads(compose_file)


@pytest.mark.parametrize("brick", _unoq_llamacpp_bricks())
def test_api_documentation_says_how_an_app_sets_the_threads(brick: str):
    descriptions = [
        item.doc.long_description or ""
        for module in sorted((BRICKS_DIR / brick).glob("*.py"))
        for item in extract_docstrings_with_types(str(module), f"arduino.app_bricks.{brick}.{module.stem}")
        if item.kind == "class" and item.doc is not None
    ]
    assert any(THREADS in description and "app.yaml" in description for description in descriptions), (
        f"the class description of the {brick} brick must say when and where an app sets {THREADS}"
    )
