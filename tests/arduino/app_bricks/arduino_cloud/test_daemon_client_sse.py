# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

import io
import threading

import pytest
import requests

from arduino.app_bricks.arduino_cloud.daemon_client import DaemonClient

STREAM = b': heartbeat\nevent: lastvalue\ndata: {"value": 1}\n\nevent: update\ndata: {"value": 2}\n\n'


@pytest.mark.parametrize("encoding", ["utf-8", None])
def test_sse_parser_handles_declared_and_missing_encoding(encoding: str | None):
    # Without a declared encoding requests yields the lines as bytes
    resp = requests.Response()
    resp.raw = io.BytesIO(STREAM)
    resp.encoding = encoding

    events = list(DaemonClient._iter_events(resp, threading.Event()))

    assert events == [("lastvalue", {"value": 1}), ("update", {"value": 2})]
