# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""Start the real server for the tests: fake memory readings and short timeouts."""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src")]
import memory  # noqa: E402
import runner  # noqa: E402

memory.available_bytes = lambda: int(os.environ.get("FAKE_MEM_MB", "8192")) << 20
# Short timeouts keep the checks quick. EI_TEST_TIMEOUT raises them on a slow board, where
# starting the fake .eim processes costs a noticeable part of the deadline.
runner.START_TIMEOUT_S = runner.REQUEST_TIMEOUT_S = int(os.environ.get("EI_TEST_TIMEOUT", "4"))
import inference_server  # noqa: E402

inference_server.OPEN_TIMEOUT_S = 1
inference_server.main()
