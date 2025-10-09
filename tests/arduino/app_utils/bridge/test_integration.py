# SPDX-FileCopyrightText: Copyright (C) 2025 ARDUINO SA <http://www.arduino.cc>
#
# SPDX-License-Identifier: MPL-2.0

import unittest
import threading
import time
import msgpack
import os
import tempfile
import socket
import queue

from unittest.mock import MagicMock, patch

from arduino.app_utils.bridge import ClientServer


class TestIntegration(unittest.TestCase):
    def setUp(self):
        """Set up for each test. Resets the singleton and creates a temporary
        directory for the Unix socket.
        """
        ClientServer._instance = None

        self.tmpdir = tempfile.TemporaryDirectory()
        self.socket_path = os.path.join(self.tmpdir.name, "test.sock")
        self.stop_server = threading.Event()
        self.server_thread = None

        # Patch dependencies
        # Mock the logger used by ClientServer
        patch("arduino.app_utils.bridge.logger", MagicMock()).start()

    def tearDown(self):
        """Clean up after each test by stopping the server thread and removing
        the temporary directory.
        """
        self.stop_server.set()

        # Make a dummy connection to unblock server.accept() if it's waiting
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
                s.settimeout(0.1)
                s.connect(self.socket_path)
        except Exception:
            pass  # This is fine, the server might already be closed.

        if self.server_thread:
            self.server_thread.join(timeout=2)

        self.tmpdir.cleanup()

    def test_notify(self):
        """Tests that ClientServer.notify correctly sends a message to the server."""
        server_ready = threading.Event()
        received_queue = queue.Queue()

        def server_logic():
            server_sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            server_sock.bind(self.socket_path)
            server_sock.listen(1)
            server_ready.set()
            conn, _ = server_sock.accept()
            unpacker = msgpack.Unpacker()
            while not self.stop_server.is_set():
                data = conn.recv(1024)
                if not data:
                    break
                unpacker.feed(data)
                for msg in unpacker:
                    received_queue.put(msg)
            conn.close()
            server_sock.close()

        self.server_thread = threading.Thread(target=server_logic, daemon=True)
        self.server_thread.start()
        self.assertTrue(server_ready.wait(timeout=2), "Server did not become ready")

        client = ClientServer(address=f"unix://{self.socket_path}")
        client._is_connected_flag.wait(timeout=2)  # Wait for client to connect

        client.notify("test_method", "hello", 123)

        try:
            received = received_queue.get(timeout=2)
            self.assertEqual(received, [2, "test_method", ["hello", 123]])
        except queue.Empty:
            self.fail("Server did not receive notify message in time.")

    def test_call(self):
        """Tests that ClientServer.call correctly sends a request and receives a response."""
        server_ready = threading.Event()

        def server_logic():
            server_sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            server_sock.bind(self.socket_path)
            server_sock.listen(1)
            server_ready.set()
            conn, _ = server_sock.accept()
            unpacker = msgpack.Unpacker(raw=False)
            data = conn.recv(1024)
            unpacker.feed(data)
            msg = next(unpacker)

            # Verify request and send response
            self.assertEqual(msg[0], 0)  # type: request
            self.assertEqual(msg[2], "get_value")
            response = [1, msg[1], None, "success!"]
            conn.sendall(msgpack.packb(response))

            self.stop_server.wait()
            conn.close()
            server_sock.close()

        self.server_thread = threading.Thread(target=server_logic, daemon=True)
        self.server_thread.start()
        self.assertTrue(server_ready.wait(timeout=2), "Server did not become ready")

        client = ClientServer(address=f"unix://{self.socket_path}")
        client._is_connected_flag.wait(timeout=2)

        result = client.call("get_value")
        self.assertEqual(result, "success!")

    def test_provide(self):
        """Tests that ClientServer.provide makes a function callable by the server."""
        server_ready = threading.Event()
        response_queue = queue.Queue()

        def server_logic():
            server_sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            server_sock.bind(self.socket_path)
            server_sock.listen(1)
            server_ready.set()
            conn, _ = server_sock.accept()
            unpacker = msgpack.Unpacker(raw=False)

            # 1. Receive $/register call from client
            data = conn.recv(1024)
            unpacker.feed(data)
            register_msg = next(unpacker)
            self.assertEqual(register_msg[0], 0)
            self.assertEqual(register_msg[2], "$/register")
            self.assertEqual(register_msg[3], ["add"])

            # 2. Send a success response for the registration
            reg_response = [1, register_msg[1], None, None]
            conn.sendall(msgpack.packb(reg_response))

            # Give the client a moment to process the registration response
            time.sleep(0.5)

            # 3. Send a request to the client to call the provided function
            call_msg = [0, 123, "add", [10, 5]]
            conn.sendall(msgpack.packb(call_msg))

            # 4. Wait for the client's response
            data = conn.recv(1024)
            unpacker.feed(data)
            response_msg = next(unpacker)
            response_queue.put(response_msg)

            self.stop_server.wait()
            conn.close()
            server_sock.close()

        self.server_thread = threading.Thread(target=server_logic, daemon=True)
        self.server_thread.start()
        self.assertTrue(server_ready.wait(timeout=2), "Server did not become ready")

        client = ClientServer(address=f"unix://{self.socket_path}")
        client._is_connected_flag.wait(timeout=2)

        client.provide("add", lambda a, b: a + b)

        try:
            final_response = response_queue.get(timeout=2)
            # [1, 123, None, 15]
            self.assertEqual(final_response[0], 1)  # type: response
            self.assertEqual(final_response[1], 123)  # msgid
            self.assertIsNone(final_response[2])  # error
            self.assertEqual(final_response[3], 15)  # result
        except queue.Empty:
            self.fail("Server did not receive response for provided method.")

    def test_reconnection(self):
        """Tests that the client automatically reconnects after the server disconnects it."""
        connections = []
        server_ready = threading.Event()

        def server_logic():
            server_sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            server_sock.bind(self.socket_path)
            server_sock.listen(1)
            server_ready.set()

            # Accept first connection and close it immediately
            conn1, _ = server_sock.accept()
            connections.append(conn1)
            conn1.close()

            # Accept the second (reconnected) connection
            conn2, _ = server_sock.accept()
            connections.append(conn2)

            self.stop_server.wait()  # Keep connection open until test ends
            conn2.close()
            server_sock.close()

        self.server_thread = threading.Thread(target=server_logic, daemon=True)
        self.server_thread.start()
        self.assertTrue(server_ready.wait(timeout=2), "Server did not become ready")

        with patch("arduino.app_utils.bridge._reconnect_delay", 0):  # Speed up reconnection for the test
            ClientServer(address=f"unix://{self.socket_path}")

            time_waited = 0
            while len(connections) < 2 and time_waited < 5:
                time.sleep(0.1)
                time_waited += 0.1

            self.assertEqual(len(connections), 2, "Client did not reconnect in time")
