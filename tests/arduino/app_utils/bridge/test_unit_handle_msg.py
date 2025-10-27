# SPDX-FileCopyrightText: Copyright (C) 2025 ARDUINO SA <http://www.arduino.cc>
#
# SPDX-License-Identifier: MPL-2.0

from unittest.mock import MagicMock

from arduino.app_utils.bridge import ClientServer, ROUTE_ALREADY_EXISTS_ERR, GENERIC_ERR
from test_unit_common import UnitTest


class TestHandleMsg(UnitTest):
    def test_empty_msg(self):
        """Test handling of an empty message."""
        client = ClientServer()
        client._handle_msg([])
        self.mock_logger.warning.assert_called_once_with("Invalid RPC message received (must be a non-empty list).")
        self.mock_logger.error.assert_not_called()

    def test_unknown_msg_type(self):
        """Test handling of an unknown message type."""
        client = ClientServer()
        client._handle_msg([99, 1, None, "result"])  # Msg type 99 does not exist
        self.mock_logger.warning.assert_called_once_with("Invalid RPC message type received: 99")
        self.mock_logger.error.assert_not_called()

    def test_unknown_msg_id(self):
        """Test handling of an unknown message id."""
        client = ClientServer()
        client._handle_msg([1, 9999, None, "result"])  # Msg id 9999 does not exist
        self.mock_logger.warning.assert_called_once_with("Response for unknown msgid 9999 received.")
        self.mock_logger.error.assert_not_called()

    def test_malformed_messages(self):
        """Test handling of malformed messages."""
        client = ClientServer()

        client._handle_msg([0, 1, "method", [0, 1], "extra field"])  # Malformed payload
        self.mock_logger.warning.assert_not_called()
        self.mock_logger.error.assert_called_once_with("Message validation error: Invalid RPC request: expected length 4, got 5")
        self.mock_logger.reset_mock()
        client._handle_msg([0, 1, "method", 1])  # Malformed params
        self.mock_logger.warning.assert_not_called()
        self.mock_logger.error.assert_called_once_with("Message validation error: Invalid RPC request params: expected array or tuple")
        self.mock_logger.reset_mock()

        client._handle_msg([1, 1, None, "result", "extra field"])  # Malformed payload
        self.mock_logger.warning.assert_not_called()
        self.mock_logger.error.assert_called_once_with("Message validation error: Invalid RPC response: expected length 4, got 5")
        self.mock_logger.reset_mock()
        client._handle_msg([1, 1, 42, "result"])  # Malformed error
        self.mock_logger.warning.assert_not_called()
        self.mock_logger.error.assert_called_once_with("Message validation error: Invalid error format in RPC response")
        self.mock_logger.reset_mock()

        client._handle_msg([2, 1, [0, 1], "extra field"])  # Malformed payload
        self.mock_logger.warning.assert_not_called()
        self.mock_logger.error.assert_called_once_with("Message validation error: Invalid RPC notification: expected length 3, got 4")
        self.mock_logger.reset_mock()
        client._handle_msg([2, 1, 42])  # Malformed params
        self.mock_logger.warning.assert_not_called()
        self.mock_logger.error.assert_called_once_with("Message validation error: Invalid RPC notification params: expected array or tuple")
        self.mock_logger.reset_mock()

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

    def test_handle_msg_request_handler_fail(self):
        """Test handling of a request for a method that fails running its handler."""
        client = ClientServer()
        client._send_response = MagicMock()

        request_msg = [0, 111, "failing_method", []]
        client.handlers["failing_method"] = MagicMock(side_effect=ValueError("Handler failed"))

        client._handle_msg(request_msg)

        client._send_response.assert_called_once()
        args, _ = client._send_response.call_args
        self.assertEqual(args[0], 111)  # msgid
        self.assertIsInstance(args[1], ValueError)  # error
        self.assertIsNone(args[2])  # result

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

    def test_handle_msg_generic_error_response(self):
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

    def test_handle_msg_method_exists_error_response(self):
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
