# SPDX-FileCopyrightText: Copyright (C) 2025 ARDUINO SA <http://www.arduino.cc>
#
# SPDX-License-Identifier: MPL-2.0

from unittest.mock import MagicMock

from arduino.app_utils.bridge import ClientServer, ROUTE_ALREADY_EXISTS_ERR, GENERIC_ERR
from test_unit_common import UnitTest


class TestHandleMsg(UnitTest):
    def test_handle_msg_request(self):
        """Test handling of an incoming request message."""
        client = ClientServer()
        client._send_response = MagicMock()

        handler_mock = MagicMock(return_value="handled")
        method_name = "provided_method"
        client.handlers[method_name] = handler_mock

        msgid = 123
        params = [1, 2, 3]
        request_msg = [0, msgid, method_name.encode(), params]  # Method name as bytes

        client._handle_msg(request_msg)

        handler_mock.assert_called_once_with(*params)
        client._send_response.assert_called_once_with(msgid, None, "handled")

    def test_handle_msg_request_method_not_found(self):
        """Test handling of a request for a method that is not found."""
        client = ClientServer()
        client._send_response = MagicMock()

        request_msg = [0, 456, "unknown_method", []]

        client._handle_msg(request_msg)

        client._send_response.assert_called_once()
        args, _ = client._send_response.call_args
        self.assertEqual(args[0], 456)  # msgid
        self.assertIsInstance(args[1], NameError)  # error
        self.assertIsNone(args[2])  # result

    def test_handle_msg_notification(self):
        """Test handling of an incoming notification message."""
        client = ClientServer()
        client._send_response = MagicMock()

        handler_mock = MagicMock()
        method_name = "notification_handler"
        client.handlers[method_name] = handler_mock

        params = ["notify", "me"]
        notification_msg = [2, method_name, params]

        client._handle_msg(notification_msg)

        handler_mock.assert_called_once_with(*params)
        client._send_response.assert_not_called()  # Notifications don't get responses

    def test_handle_msg_response(self):
        """Test handling of an incoming response message."""
        client = ClientServer()

        msgid = 789
        result_data = {"status": "ok"}

        # Mock the callbacks
        on_result_mock = MagicMock()
        on_error_mock = MagicMock()
        client.callbacks[msgid] = (on_result_mock, on_error_mock)

        response_msg = [1, msgid, None, result_data]

        client._handle_msg(response_msg)

        on_result_mock.assert_called_once_with(result_data)
        on_error_mock.assert_not_called()
        self.assertNotIn(msgid, client.callbacks)  # Callback should be removed

    def test_handle_msg_response_function_not_found(self):
        """Test handling of an incoming error response message."""
        client = ClientServer()

        msgid = 101112
        result_data = None
        result_error = [GENERIC_ERR, "Some generic error occurred"]

        # Mock the callbacks
        on_result_mock = MagicMock()
        on_error_mock = MagicMock()
        client.callbacks[msgid] = (on_result_mock, on_error_mock)

        response_msg = [1, msgid, result_error, result_data]

        client._handle_msg(response_msg)

        on_result_mock.assert_not_called()
        on_error_mock.assert_called_once_with(result_error)
        self.assertNotIn(msgid, client.callbacks)  # Callback should be removed

    def test_handle_msg_response_already_provided_error(self):
        """Test handling of an incoming error response message that signals a method is already provided."""
        client = ClientServer()

        msgid = 131415
        result_data = None
        result_error = [ROUTE_ALREADY_EXISTS_ERR, "Method already exists"]

        # Mock the callbacks
        on_result_mock = MagicMock()
        on_error_mock = MagicMock()
        client.callbacks[msgid] = (on_result_mock, on_error_mock)

        response_msg = [1, msgid, result_error, result_data]

        client._handle_msg(response_msg)

        on_result_mock.assert_called_once_with(result_data)
        on_error_mock.assert_not_called()
        self.assertNotIn(msgid, client.callbacks)  # Callback should be removed
