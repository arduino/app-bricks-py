# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

import os
import socket
import tempfile
import threading
import unittest
from unittest.mock import MagicMock, patch

from arduino.app_utils.bridge import _ClientServer


class TestLifecycle(unittest.TestCase):
    """Lifecycle tests for the connection worker, using real threads."""

    def setUp(self):
        self.logger_patcher = patch("arduino.app_utils.bridge.logger", MagicMock())
        self.logger_patcher.start()
        self.addCleanup(self.logger_patcher.stop)

    def _start_dummy_server(self):
        """Starts a Unix-socket server that accepts one connection and holds it open."""
        tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(tmpdir.cleanup)
        sock_path = os.path.join(tmpdir.name, "test.sock")

        server_sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        server_sock.bind(sock_path)
        server_sock.listen(1)
        self.addCleanup(server_sock.close)

        ready = threading.Event()

        def serve():
            ready.set()
            try:
                conn, _ = server_sock.accept()
                self.addCleanup(conn.close)
            except OSError:
                pass

        threading.Thread(target=serve, daemon=True).start()
        self.assertTrue(ready.wait(timeout=2), "Dummy server did not become ready")
        return sock_path

    def test_start_then_stop_joins_background_thread(self):
        """start() spawns a real background thread; stop() must join it so it does not leak."""
        sock_path = self._start_dummy_server()

        client = _ClientServer(address=f"unix://{sock_path}")
        client.start()
        self.assertTrue(client._is_connected_flag.wait(timeout=2), "Client did not connect")
        background_thread = client._read_thread
        self.assertTrue(background_thread.is_alive())

        client.stop()

        self.assertFalse(background_thread.is_alive(), "Background thread leaked after stop()")
        self.assertIsNone(client._read_thread)

    def test_stop_without_start_is_safe(self):
        """stop() must be a safe no-op even if start() was never called."""
        client = _ClientServer(address="unix:///tmp/never-exists.sock")
        client.stop()  # Must not raise or block
        self.assertIsNone(client._read_thread)

    def test_context_manager_starts_and_stops(self):
        """The worker can be used as a context manager that starts on enter and stops on exit."""
        client = _ClientServer(address="unix:///tmp/never-exists.sock")

        # Avoid real connecting/looping
        with patch.object(_ClientServer, "_connect"), patch.object(_ClientServer, "_conn_manager", lambda self: self._stop_event.wait()):
            with client as entered:
                self.assertIs(entered, client)
                self.assertTrue(client._read_thread.is_alive())

        self.assertTrue(client._stop_event.is_set())
        self.assertFalse(client._read_thread is not None and client._read_thread.is_alive())

    def test_start_is_idempotent(self):
        """Calling start() twice does not spawn a second background thread."""
        with patch.object(_ClientServer, "_connect"), patch.object(_ClientServer, "_conn_manager", lambda self: self._stop_event.wait()):
            client = _ClientServer(address="unix:///tmp/never-exists.sock")
            client.start()
            first_thread = client._read_thread
            client.start()  # idempotent
            self.assertIs(client._read_thread, first_thread)
            client.stop()
