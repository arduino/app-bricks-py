# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

import unittest
from unittest.mock import MagicMock, patch

from arduino.app_utils.bridge import ClientServer


class UnitTest(unittest.TestCase):
    def setUp(self):
        """This method is called before each test to reset the singleton and patch the dependencies."""
        ClientServer._instance = None

        # Patch dependencies
        # Mock the logger used by ClientServer
        self.mock_logger = MagicMock()
        self.logger_patcher = patch("arduino.app_utils.bridge.logger", self.mock_logger)
        self.logger_patcher.start()

        # Mock the socket instance that will be created
        self.mock_socket_instance = MagicMock()
        self.socket_patcher = patch("arduino.app_utils.bridge.socket")
        self.mock_socket = self.socket_patcher.start()
        self.mock_socket.socket.return_value = self.mock_socket_instance
        self.mock_socket.create_connection.return_value = self.mock_socket_instance

        # Mock only threading.Thread so the background read loop never runs.
        self.mock_thread_instance = MagicMock()
        self.thread_patcher = patch("arduino.app_utils.bridge.threading.Thread", return_value=self.mock_thread_instance)
        self.mock_thread = self.thread_patcher.start()

    def tearDown(self):
        """This method is called after each test and cleans up the patched dependencies."""
        instance = ClientServer._instance
        if instance is not None:
            instance.stop()
        ClientServer._instance = None

        self.thread_patcher.stop()
        self.socket_patcher.stop()
        self.logger_patcher.stop()
