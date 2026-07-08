# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

import pytest
from langchain_ollama import ChatOllama


@pytest.fixture(autouse=True)
def model_factory_fixture(monkeypatch):
    import arduino.app_bricks.cloud_llm.cloud_llm as cloud_llm

    real_model_factory = cloud_llm.model_factory

    def _model_factory(model_name: str, **kwargs):
        if model_name.startswith("ollama"):
            base_url = kwargs.pop("base_url", "http://localhost:11434")
            name = model_name.split("-", 1)[1]
            return ChatOllama(base_url=base_url, model=name)

        return real_model_factory(model_name, **kwargs)  # type: ignore

    monkeypatch.setattr(cloud_llm, "model_factory", _model_factory)
    yield
