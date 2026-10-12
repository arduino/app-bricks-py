# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""The WebUI import shortcuts (app start time) skip work the app does not need and leave everything else behaving as usual."""

import subprocess
import sys
import textwrap

from arduino.app_bricks.web_ui._fast_imports import defer_socketio_client_dependencies


def _run(code: str) -> None:
    """Run code in a fresh interpreter, where the WebUI import is the first one to load FastAPI."""
    result = subprocess.run([sys.executable, "-c", textwrap.dedent(code)], capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr


def test_webui_import_does_not_load_the_socketio_sync_client_dependencies():
    _run("""
        import sys
        import arduino.app_bricks.web_ui
        import engineio.client

        assert "requests" not in sys.modules, "requests loaded by the WebUI import"
        assert "websocket" not in sys.modules, "websocket-client loaded by the WebUI import"
        assert engineio.client.requests is not None and engineio.client.websocket is not None

        import requests  # A real import still gets the real module, not a leftover None
        assert engineio.client.requests.Session is requests.Session
    """)


def test_socketio_sync_client_still_works_after_webui_import():
    _run("""
        import socket
        import arduino.app_bricks.web_ui
        import socketio

        with socket.socket() as s:  # A port the OS just handed out and nobody listens on
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]

        # A closed port: the client must get as far as the HTTP request, not trip on a missing requests
        try:
            socketio.Client().connect(f"http://127.0.0.1:{port}", wait_timeout=1, transports=["polling"])
        except socketio.exceptions.ConnectionError as e:
            assert "HTTPConnectionPool" in str(e) or "Connection refused" in str(e), e
        else:
            raise AssertionError("connecting to a closed port succeeded")
    """)


def test_missing_optional_dependency_is_left_alone():
    with defer_socketio_client_dependencies(names=["arduino_no_such_module"]):
        assert "arduino_no_such_module" not in sys.modules
    assert "arduino_no_such_module" not in sys.modules


def test_hidden_dependency_is_importable_again_after_the_block():
    name = "colorsys"  # Stdlib, rarely preloaded: stands in for requests without touching it
    sys.modules.pop(name, None)
    with defer_socketio_client_dependencies(names=[name]):
        try:
            __import__(name)
        except ImportError:
            pass
        else:
            raise AssertionError(f"{name} was importable inside the block")
    assert __import__(name).__name__ == name


def test_modules_imported_in_the_block_get_a_stand_in_for_a_hidden_dependency(tmp_path, monkeypatch):
    """Not only engineio: any optional import of a hidden module inside the block ends up working."""
    (tmp_path / "optional_user.py").write_text("try:\n    import colorsys\nexcept ImportError:\n    colorsys = None\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    for name in ("colorsys", "optional_user"):
        monkeypatch.delitem(sys.modules, name, raising=False)

    with defer_socketio_client_dependencies(names=["colorsys"]):
        import optional_user  # pyright: ignore[reportMissingImports]

        assert optional_user.colorsys is None

    assert "colorsys" not in sys.modules, "the hidden module must not be imported by the fix-up itself"
    assert optional_user.colorsys.rgb_to_hsv(1.0, 0.0, 0.0) == (0.0, 1.0, 1.0)
