# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""Guards the llama.cpp runners against core dumps.

The runners work in /models, the host's models directory, and the boards write a core into the
working directory of the process that crashed (core_pattern "core"). Two llama-server aborts on
21linux (2026-11-24, fastrpc_mmap failed at load) left 2.6 and 4.3 GB there.
"""

from pathlib import Path

import pytest
import yaml

SERVICE_DIR = Path(__file__).resolve().parents[3] / "src" / "arduino" / "app_services" / "llamacpp"


@pytest.mark.parametrize("compose_file", sorted(SERVICE_DIR.glob("service_compose*.yaml")), ids=lambda path: path.name)
def test_the_runner_dumps_no_core(compose_file: Path):
    services = yaml.safe_load(compose_file.read_text(encoding="utf-8"))["services"]

    for name, service in services.items():
        assert service.get("ulimits", {}).get("core") == 0, name
