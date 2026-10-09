# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""The WebUI import shortcuts (app start time) skip work the app does not need and leave everything else behaving as usual."""

import gc
import subprocess
import sys
import textwrap

from pydantic import BaseModel
from pydantic._internal._config import config_defaults

from arduino.app_bricks.web_ui._fast_imports import defer_pydantic_model_builds, defer_socketio_client_dependencies, gc_paused


def _run(code: str) -> None:
    """Run code in a fresh interpreter, where the WebUI import is the first one to load FastAPI."""
    result = subprocess.run([sys.executable, "-c", textwrap.dedent(code)], capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr


def test_webui_import_leaves_fastapi_openapi_models_unbuilt():
    _run("""
        import arduino.app_bricks.web_ui
        from fastapi.openapi import models
        from pydantic._internal._config import config_defaults

        assert models.Contact.__pydantic_complete__ is False, "FastAPI OpenAPI models were built at import"
        assert config_defaults["defer_build"] is False, "the pydantic default leaked out of the WebUI import"
    """)


def test_deferred_fastapi_models_still_work_on_first_use():
    _run("""
        import arduino.app_bricks.web_ui
        from fastapi.openapi import models

        spec = models.OpenAPI.model_validate({"openapi": "3.1.0", "info": {"title": "t", "version": "1"}, "paths": {}})
        assert spec.info.title == "t"
        assert models.Contact.model_validate({"name": "x"}).name == "x"
    """)


def test_app_models_and_routes_behave_as_usual_after_webui_import():
    _run("""
        from fastapi.testclient import TestClient
        from pydantic import BaseModel
        from arduino.app_bricks.web_ui import WebUI

        class Item(BaseModel):
            name: str
            count: int = 1

        assert Item.__pydantic_complete__ is True, "app models must still be built at definition"

        ui = WebUI()
        ui.expose_api("GET", "/double", lambda x: {"x": x * 2})  # Untyped query param: a string

        def create(item: Item) -> Item:
            return item

        ui.expose_api("POST", "/typed", create)
        client = TestClient(ui.app)
        assert client.post("/typed", json={"name": "a", "count": 3}).json() == {"name": "a", "count": 3}
        assert client.post("/typed", json={"count": "nope"}).status_code == 422
        assert client.get("/double", params={"x": 21}).json() == {"x": "2121"}
    """)


def test_context_manager_restores_default_on_error():
    previous = config_defaults.get("defer_build", False)
    try:
        with defer_pydantic_model_builds():
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    assert config_defaults.get("defer_build", False) == previous

    class Built(BaseModel):
        value: int

    assert Built.__pydantic_complete__ is True


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


def test_webui_import_leaves_the_garbage_collector_enabled():
    _run("""
        import gc
        import arduino.app_bricks.web_ui
        assert gc.isenabled()
    """)


def test_gc_paused_restores_the_previous_state():
    assert gc.isenabled()
    with gc_paused():
        assert not gc.isenabled()
    assert gc.isenabled()

    gc.disable()
    try:
        with gc_paused():
            pass
        assert not gc.isenabled(), "an app that disabled the collector must keep it disabled"
    finally:
        gc.enable()

    try:
        with gc_paused():
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    assert gc.isenabled()
