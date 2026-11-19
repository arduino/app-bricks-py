# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

import json

import pytest

from arduino.app_tools.app_launcher.protocol import ErrorCode, LauncherError, Start, Status, decode_request, encode, error, ok


def test_decodes_every_request():
    assert decode_request(b'{"cmd":"start"}\n') == Start()
    assert decode_request(b'{"cmd":"start","env":{"K":"V","EMPTY":""}}') == Start(env={"K": "V", "EMPTY": ""})
    assert decode_request(b'{"cmd":"start","app":"/other"}') == Start(), "a field the launcher does not know is ignored: it runs its own app"
    assert decode_request(b'{"cmd":"status"}') == Status()


@pytest.mark.parametrize(
    "line",
    [
        b"not json",
        b"\xff",
        b"[1, 2]",
        b'{"app":"/apps/a"}',
        b'{"cmd":"reboot"}',
        b'{"cmd":"stop"}',
        b'{"cmd":"start","env":["K=V"]}',
        b'{"cmd":"start","env":{"K":1}}',
        b'{"cmd":"start","env":{"K=X":"V"}}',
        b'{"cmd":"start","env":{"":"V"}}',
        b'{"cmd":"start","env":{"K":"V\\u0000"}}',
    ],
)
def test_rejects_malformed_requests(line):
    with pytest.raises(LauncherError) as raised:
        decode_request(line)
    assert raised.value.code is ErrorCode.BAD_REQUEST


def test_responses_are_single_json_lines():
    line = encode(ok(pid=12, timings={"stop_ms": 1.5}))
    assert line.endswith(b"\n") and line.count(b"\n") == 1
    assert json.loads(line) == {"ok": True, "pid": 12, "timings": {"stop_ms": 1.5}}
    assert json.loads(encode(error(LauncherError(ErrorCode.BUSY, "shutting down")))) == {
        "ok": False,
        "error": {"code": "busy", "message": "shutting down"},
    }
