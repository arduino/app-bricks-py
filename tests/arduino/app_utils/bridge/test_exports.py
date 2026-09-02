# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""Validates that arduino.app_utils exposes the app-wide bridge API and configures the arduino-router-bridge library for the app context."""

import io
import logging
import os
import socket
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr
from unittest.mock import patch

from arduino.app_utils.logger import _build_handler, _configure_library_logger


class TestBridgeExports(unittest.TestCase):
    def test_bridge_api_is_exported(self):
        from arduino.app_utils import Bridge, call, notify, provide
        from arduino.app_utils import bridge as bridge_module

        self.assertIs(Bridge, bridge_module.Bridge)
        self.assertIs(notify, bridge_module.notify)
        self.assertIs(call, bridge_module.call)
        self.assertIs(provide, bridge_module.provide)

    def test_library_logger_is_configured_on_import(self):
        import arduino.app_utils  # noqa: F401  Importing the package configures the library logger

        lib_logger = logging.getLogger("arduino.router_bridge")
        # Other tests and third-party libraries may attach their own handlers to this
        # process-global logger, so only assert on the app-standard one being in place.
        app_format = _build_handler().formatter._fmt
        app_handlers = [h for h in lib_logger.handlers if h.formatter is not None and h.formatter._fmt == app_format]
        foreign_handlers = [type(h).__name__ for h in lib_logger.handlers]
        self.assertEqual(len(app_handlers), 1, f"expected exactly one app-standard handler among: {foreign_handlers}")
        self.assertFalse(lib_logger.propagate)

    def test_app_socket_env_is_applied_to_bridge(self):
        """An app importing arduino.app_utils connects to the router address set via APP_SOCKET."""
        with tempfile.TemporaryDirectory() as tmpdir:
            socket_path = os.path.join(tmpdir, "router.sock")
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as server:
                server.bind(socket_path)
                server.listen(1)
                server.settimeout(30)

                code = "import arduino.app_utils; import time; time.sleep(30)"  # Stay alive while the bridge connects in the background
                app = subprocess.Popen([sys.executable, "-c", code], env={**os.environ, "APP_SOCKET": f"unix://{socket_path}"})
                try:
                    connection, _ = server.accept()  # Times out and fails if the app connects elsewhere
                    connection.close()
                finally:
                    app.terminate()
                    app.wait(timeout=10)


class TestConfigureLibraryLogger(unittest.TestCase):
    def _configure_and_log(self, **kwargs):
        stderr = io.StringIO()
        with redirect_stderr(stderr):  # The handler binds the redirected stream at configuration time
            _configure_library_logger("some.test.lib", **kwargs)
            logging.getLogger("some.test.lib.child").info("hello from lib")
        return stderr.getvalue()

    def test_applies_app_format(self):
        output = self._configure_and_log()
        self.assertIn("INFO", output)
        self.assertIn("some.test.lib.child:  hello from lib", output)

    def test_display_name_replaces_logger_name(self):
        output = self._configure_and_log(display_name="Nice")
        self.assertIn("Nice:  hello from lib", output)
        self.assertNotIn("some.test.lib", output)

    def test_honors_app_bricks_log_level(self):
        with patch.dict("os.environ", {"APP_BRICKS_LOG_LEVEL": "ERROR"}):
            output = self._configure_and_log()
        self.assertEqual(output, "")


if __name__ == "__main__":
    unittest.main()
