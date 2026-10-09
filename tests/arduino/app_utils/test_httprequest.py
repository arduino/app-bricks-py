# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

import json
import subprocess
import sys
from typing import Any

import pytest
import requests
from requests.adapters import HTTPAdapter

from arduino.app_utils.httprequest import HttpClient


def _loaded_after(code: str, modules: list[str]) -> dict[str, bool]:
    """Run code in a fresh interpreter and report which of the given modules it left in sys.modules."""
    probe = f"{code}\nimport json, sys\nprint(json.dumps({{m: m in sys.modules for m in {modules!r}}}))"
    result = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, timeout=120, check=True)
    return json.loads(result.stdout.strip().splitlines()[-1])


@pytest.mark.parametrize("code", ["import arduino.app_utils", "from arduino.app_utils import HttpClient"])
def test_import_does_not_load_requests(code: str):
    assert _loaded_after(code, ["requests", "urllib3"]) == {"requests": False, "urllib3": False}


def test_session_adapter_uses_the_configured_retries():
    client = HttpClient(total_retries=3, backoff_factor=2, status_forcelist=(503,))
    session: requests.Session = getattr(client, "_HttpClient__http_session")

    for url in ("http://example.com", "https://example.com"):
        adapter = session.get_adapter(url)
        assert isinstance(adapter, HTTPAdapter)
        retries = adapter.max_retries
        assert (retries.total, retries.connect, retries.read) == (3, 3, 3)
        assert retries.backoff_factor == 2
        assert retries.status_forcelist == [503]
    client.close()


def test_request_with_retry_returns_none_on_request_exception(monkeypatch: pytest.MonkeyPatch):
    client = HttpClient()
    session: requests.Session = getattr(client, "_HttpClient__http_session")

    def fail(*args: Any, **kwargs: Any) -> requests.Response:
        raise requests.exceptions.ConnectionError("unreachable")

    monkeypatch.setattr(session, "request", fail)
    assert client.request_with_retry("http://example.com") is None
    client.close()


def test_request_with_retry_returns_none_on_empty_url():
    client = HttpClient()
    assert client.request_with_retry("") is None
    client.close()
