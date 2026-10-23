# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

import json

import pytest

from arduino.app_tools.app_launcher.protocol import ErrorCode, LauncherError, Start, Status, Stop, decode_request, encode, error, ok


def test_decodes_every_request():
    assert decode_request(b'{"cmd":"start","app":"/app/"}\n') == Start(app="/app")
    assert decode_request(b'{"cmd":"start","app":"/app","env":{"K":"V","EMPTY":""}}') == Start(app="/app", env={"K": "V", "EMPTY": ""})
    assert decode_request(b'{"cmd":"stop"}') == Stop()
    assert decode_request(b'{"cmd":"stop","app":"/app"}') == Stop(), "one app per container, the name is ignored"
    assert decode_request(b'{"cmd":"status"}') == Status()


@pytest.mark.parametrize(
    "line",
    [
        b"not json",
        b"\xff",
        b"[1, 2]",
        b'{"app":"/apps/a"}',
        b'{"cmd":"reboot"}',
        b'{"cmd":"start"}',
        b'{"cmd":"start","app":"apps/a"}',
        b'{"cmd":"start","app":"/apps/a","env":["K=V"]}',
        b'{"cmd":"start","app":"/apps/a","env":{"K":1}}',
        b'{"cmd":"start","app":"/apps/a","env":{"K=X":"V"}}',
        b'{"cmd":"start","app":"/apps/a","env":{"":"V"}}',
        b'{"cmd":"start","app":"/apps/a","env":{"K":"V\\u0000"}}',
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
