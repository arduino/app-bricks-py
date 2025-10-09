# SPDX-FileCopyrightText: Copyright (C) 2025 ARDUINO SA <http://www.arduino.cc>
#
# SPDX-License-Identifier: MPL-2.0

from unittest.mock import MagicMock

from arduino.app_utils.bridge import ClientServer, GENERIC_ERR
from test_unit_common import UnitTest


class TestErrors(UnitTest):
    def test_connection_lost(self):
        """Test that pending callbacks fail and are cleaned up when connection is lost."""
        client = ClientServer()

        on_error_1 = MagicMock()
        on_error_2 = MagicMock()

        client.callbacks[1] = (None, on_error_1)
        client.callbacks[2] = (None, on_error_2)

        reason = ConnectionError("Connection to router lost.")
        client._fail_pending_callbacks(reason)  # This call is triggered by ConnectionResetError

        on_error_1.assert_called_once_with(reason)
        on_error_2.assert_called_once_with(reason)
        self.assertEqual(len(client.callbacks), 0)

    def test_call_timeout(self):
        """Test that an RPC call raises a TimeoutError if no response is received."""
        client = ClientServer()
        client._send_bytes = MagicMock()  # Don't simulate a response

        with self.assertRaises(TimeoutError):
            client.call("test_timeout", timeout=0.1)

    def test_call_server_error(self):
        """Test an RPC call that returns an error from the server."""
        client = ClientServer()
        client._send_bytes = MagicMock()

        method_name = "test_error"
        error_response = [GENERIC_ERR, "Something went wrong"]
        msgid = client.next_msgid + 1

        def side_effect(*args, **kwargs):
            _, on_error = client.callbacks[msgid]
            on_error(error_response)

        client._send_bytes.side_effect = side_effect

        with self.assertRaises(ValueError) as cm:
            client.call(method_name)

        self.assertIn("Something went wrong", str(cm.exception))

    def test_provide_error(self):
        """Test that providing a non-callable handler raises a ValueError."""
        client = ClientServer()
        with self.assertRaises(ValueError):
            client.provide("bad_handler", "not a function")

    def test_clear_callbacks_after_connection_lost(self):
        """Test that pending callbacks are correctly failed when the connection is lost."""
        client = ClientServer()

        on_error_1 = MagicMock()
        on_error_2 = MagicMock()

        client.callbacks[1] = (None, on_error_1)
        client.callbacks[2] = (None, on_error_2)

        reason = ConnectionError("Connection to router lost.")
        client._fail_pending_callbacks(reason)  # This call is triggered by ConnectionResetError in _read_loop

        on_error_1.assert_called_once_with(reason)
        on_error_2.assert_called_once_with(reason)
        self.assertEqual(len(client.callbacks), 0)
