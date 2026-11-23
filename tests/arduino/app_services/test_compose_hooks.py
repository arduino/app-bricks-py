# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""Guards the lifecycle hooks of the compose files against a lost $$ escape.

arduino-app-cli copies each service and brick compose into the app and interpolates it twice: the
first pass turns an escaped $$ into $, the second substitutes it. A hook using $ then runs with
blanks: the Genie runner hook ran `kill -TERM ` and the runner ended in SIGKILL (exit 137).
"""

from pathlib import Path

import pytest
import yaml

SRC = Path(__file__).resolve().parents[3] / "src" / "arduino"
COMPOSE_FILES = sorted([*SRC.glob("app_services/**/service_compose*.yaml"), *SRC.glob("app_bricks/**/brick_compose*.yaml")])


@pytest.mark.parametrize("compose_file", COMPOSE_FILES, ids=lambda path: str(path.relative_to(SRC)))
def test_hooks_use_no_dollar(compose_file: Path):
    services = (yaml.safe_load(compose_file.read_text()) or {}).get("services", {})
    for name, service in services.items():
        for hook in [*service.get("pre_stop", []), *service.get("post_start", [])]:
            command = hook["command"]
            assert "$" not in (command if isinstance(command, str) else " ".join(command)), f"{name}: {command}"
