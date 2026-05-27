# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import json
import glob
from typing import cast, Callable
import os
import time
import threading
import requests
import uvicorn
from logging import Logger


from arduino.app_bricks.cloud_llm import CloudModelProvider, CloudModel
from stubs.granular_mcp_server import granular_mcp_server_factory
from stubs.object_mcp_server import object_mcp_server_factory

from deepeval.dataset import EvaluationDataset, Golden
from deepeval.test_case import LLMTestCase, ToolCall
from fastapi import FastAPI


logger = Logger("cloud_llm_tests")
pytest_plugins = ["fixtures.judge_model", "fixtures.runners", "fixtures.model_factory"]


@dataclass(frozen=True)
class ModelConfig:
    name: CloudModel | str
    provider: str
    requires_api_key: bool = True
    api_key: str | None = None
    base_url: str | None = None


models_to_test = [
    ModelConfig(name="gemini-2.5-flash", provider=CloudModelProvider.GOOGLE, api_key=os.getenv("GEMINI_API_KEY")),
    ModelConfig(
        name="ollama-qwen2.5:7b",
        provider="ollama",
        requires_api_key=False,
    ),
]


def _load_cases() -> list[dict]:
    cases: list[dict] = []
    paths = sorted(p for p in glob.glob("tests/arduino/app_bricks/cloud_llm/cases/*.json") if not os.path.basename(p).startswith("_"))
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            c = json.load(f)
            if isinstance(c, list):
                cases.extend(c)
            else:
                cases.append(c)
    return cases


def load_dataset() -> EvaluationDataset:
    goldens = [
        Golden(
            input=raw["prompt"],
            multimodal=False,
            expected_tools=[
                ToolCall(name=tool["name"], input_parameters=tool.get("input_parameters", {})) for tool in raw.get("must_call_tools", [])
            ],
            additional_metadata={
                "case_id": raw.get("id"),
                "runners": raw.get("runners", []),
                "semantic": raw.get("semantic", None),
            },
        )
        for raw in _load_cases()
    ]

    dataset = EvaluationDataset(goldens=goldens)

    for golden in dataset.goldens:
        golden = cast(Golden, golden)
        golden.additional_metadata = cast(dict, golden.additional_metadata)

        case_id = golden.additional_metadata.get("case_id", None)
        if case_id is None:
            logger.warning("Golden missing case_id: %s", golden)
            continue

        semantic_test = golden.additional_metadata.get("semantic", None)

        # Combinations of runners and models
        for runner_name in golden.additional_metadata.get("runners", []):
            for model in models_to_test:
                test_case = LLMTestCase(
                    name=(f"[bold]parent_case_id:[/bold] {case_id}\n[bold]runner:[/bold] {runner_name}\n[bold]model:[/bold] {model.name}"),
                    input=golden.input,
                    expected_tools=golden.expected_tools,
                    additional_metadata={
                        "runner_name": runner_name,
                        "model": model,
                        "semantic": semantic_test,
                    },
                )
                dataset.add_test_case(test_case)

    return dataset


# MCP server management for tests
def _is_worker() -> bool:
    return os.environ.get("PYTEST_XDIST_WORKER") is not None


def _is_mcp_srv_ready(port: int) -> bool:
    try:
        r = requests.get(f"http://127.0.0.1:{port}/health", timeout=0.3)
        return r.status_code == 200
    except requests.RequestException:
        return False


def _wait_mcp_srv_ready(port: int, timeout: float = 10.0) -> None:
    start = time.time()
    while time.time() - start < timeout:
        if _is_mcp_srv_ready(port):
            return
        time.sleep(0.1)
    raise TimeoutError(f"MCP server at 127.0.0.1:{port} not ready")


def _wait_mcp_srvs_ready(ports: list[int], timeout: float = 10.0) -> None:
    with ThreadPoolExecutor(max_workers=len(ports)) as executor:
        futures = [executor.submit(_wait_mcp_srv_ready, port, timeout) for port in ports]
        for future in futures:
            future.result()


def _start_mcp_srv(port: int, factory: Callable[[int], FastAPI]) -> None:
    app = factory(port)
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(config)
    threading.Thread(target=server.run, daemon=True).start()


def pytest_sessionstart(session):
    mcp_srvs_port_factory_map = {
        8000: granular_mcp_server_factory,
        8001: object_mcp_server_factory,
    }

    if not _is_worker():
        for port, factory in mcp_srvs_port_factory_map.items():
            _start_mcp_srv(port, factory)

    _wait_mcp_srvs_ready(list(mcp_srvs_port_factory_map))

    logger.info("MCP servers are ready")
