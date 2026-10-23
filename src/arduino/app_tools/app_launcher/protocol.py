# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""Control socket protocol: one JSON object per line, one request and one response per connection."""

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, cast

DEFAULT_SOCKET = "/run/arduino-app-cli/app-launcher.sock"
MAX_LINE_BYTES = 1024 * 1024


class ErrorCode(StrEnum):
    BAD_REQUEST = "bad_request"
    NOT_FOUND = "not_found"
    BUSY = "busy"
    STALE = "stale"
    """The app dependencies changed since this container prepared them: the app needs a fresh container."""
    INTERNAL = "internal"


class LauncherError(Exception):
    """A request the launcher refuses or fails, reported to the client with its code."""

    def __init__(self, code: ErrorCode, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass(frozen=True)
class Start:
    app: str
    env: Mapping[str, str] = field(default_factory=dict[str, str])
    """The app environment, laid over the container one."""


@dataclass(frozen=True)
class Stop:
    pass


@dataclass(frozen=True)
class Status:
    pass


Request = Start | Stop | Status


def encode(message: dict[str, Any]) -> bytes:
    """Serializes a message as one line."""
    return json.dumps(message, separators=(",", ":")).encode() + b"\n"


def ok(**fields: Any) -> dict[str, Any]:
    """Builds a success response."""
    return {"ok": True, **fields}


def error(err: LauncherError) -> dict[str, Any]:
    """Builds an error response."""
    return {"ok": False, "error": {"code": str(err.code), "message": err.message}}


def decode_request(line: bytes) -> Request:
    """Parses and validates one request line.

    Raises:
        LauncherError: bad_request when the line is not a valid request.
    """
    try:
        message: Any = json.loads(line)
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        raise LauncherError(ErrorCode.BAD_REQUEST, f"invalid JSON: {e}") from None
    match message:
        case {"cmd": "start"}:
            return Start(app=_app(message.get("app")), env=_env(message.get("env", {})))
        case {"cmd": "stop"}:
            # One app per container: an app named in the request is ignored
            return Stop()
        case {"cmd": "status"}:
            return Status()
        case {"cmd": cmd}:
            raise LauncherError(ErrorCode.BAD_REQUEST, f"unknown cmd: {cmd!r}")
        case _:
            raise LauncherError(ErrorCode.BAD_REQUEST, "a request must be a JSON object with a cmd")


def _app(value: object) -> str:
    match value:
        case str() if os.path.isabs(value):
            return os.path.normpath(value)
        case _:
            raise LauncherError(ErrorCode.BAD_REQUEST, "app must be the absolute path of the app folder")


def _env(value: object) -> dict[str, str]:
    match value:
        case dict():
            env: dict[str, str] = {}
            for item in cast(dict[object, object], value).items():
                match item:
                    case (str() as key, str() as text) if key and "=" not in key and "\0" not in key + text:
                        env[key] = text
                    case _:
                        raise LauncherError(ErrorCode.BAD_REQUEST, "env must map variable names, without = or NUL, to strings")
            return env
        case _:
            raise LauncherError(ErrorCode.BAD_REQUEST, "env must be an object of strings")
