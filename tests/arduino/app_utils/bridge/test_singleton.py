# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""Validates the app-wide singleton layer built on top of the instance-based arduino-router-bridge library."""

import unittest
from unittest.mock import MagicMock, patch

from arduino.router_bridge import DEFAULT_ADDRESS

from arduino.app_utils import bridge as bridge_module
from arduino.app_utils.bridge import Bridge, call, notify, provide


class SingletonTestCase(unittest.TestCase):
    def setUp(self):
        """Resets the shared instance and replaces the library class so no real connection is created."""
        self._reset_shared_instance()
        self.addCleanup(self._reset_shared_instance)

        self.mock_instance = MagicMock()
        router_bridge_patcher = patch.object(bridge_module, "RouterBridge", return_value=self.mock_instance)
        self.mock_router_bridge = router_bridge_patcher.start()
        self.addCleanup(router_bridge_patcher.stop)

    def _reset_shared_instance(self):
        with bridge_module._bridge_lock:
            bridge_module._bridge = None


class TestSharedInstance(SingletonTestCase):
    def test_instance_is_created_once_and_shared(self):
        first = bridge_module._get_bridge()
        second = bridge_module._get_bridge()

        self.assertIs(first, second)
        self.mock_router_bridge.assert_called_once()

    def test_every_access_reconnects_a_stopped_bridge(self):
        bridge_module._get_bridge()
        bridge_module._get_bridge()

        self.assertEqual(self.mock_instance.connect.call_count, 2)  # connect() is idempotent on a running bridge

    def test_app_socket_env_selects_the_address(self):
        with patch.dict("os.environ", {"APP_SOCKET": "unix:///tmp/app.sock"}):
            bridge_module._get_bridge()

        self.mock_router_bridge.assert_called_once_with("unix:///tmp/app.sock")

    def test_default_address_without_app_socket(self):
        with patch.dict("os.environ", clear=True):
            bridge_module._get_bridge()

        self.mock_router_bridge.assert_called_once_with(DEFAULT_ADDRESS)


class TestBridgeFacade(SingletonTestCase):
    def test_facade_delegates_to_the_shared_instance(self):
        Bridge.notify("a_method", 1, 2)
        self.mock_instance.notify.assert_called_once_with("a_method", 1, 2)

        self.assertEqual(Bridge.call("b_method", 3, timeout=5), self.mock_instance.call.return_value)
        self.mock_instance.call.assert_called_once_with("b_method", 3, timeout=5)

        handler = lambda: None
        Bridge.provide("c_method", handler)
        self.mock_instance.provide.assert_called_once_with("c_method", handler)

        Bridge.unprovide("c_method")
        self.mock_instance.unprovide.assert_called_once_with("c_method")


class TestDecorators(SingletonTestCase):
    def test_notify_and_call_decorators_are_lazy(self):
        """Decorating with @notify/@call must not create the shared instance."""

        @notify()
        def set_led(color: str, status: bool): ...

        @call("math.add", timeout=3)
        def add(a: int, b: int) -> int: ...

        self.assertIsNone(bridge_module._bridge)

    def test_notify_decorator_sends_notification_on_invocation(self):
        @notify("custom.name")
        def set_led(color: str, status: bool): ...

        set_led("green", True)
        self.mock_instance.notify.assert_called_once_with("custom.name", "green", True)

        with self.assertRaises(TypeError):
            set_led(color="green", status=True)  # Keyword args are not supported

    def test_call_decorator_calls_on_invocation(self):
        @call(timeout=3)
        def add(a: int, b: int) -> int: ...

        self.assertEqual(add(1, 2), self.mock_instance.call.return_value)
        self.mock_instance.call.assert_called_once_with("add", 1, 2, timeout=3)
        self.mock_instance.call.reset_mock()

        add(1, 2, timeout=7)  # Per-invocation override
        self.mock_instance.call.assert_called_once_with("add", 1, 2, timeout=7)

        with self.assertRaises(TypeError):
            add(a=1, b=2)  # Keyword args are not supported

    def test_provide_decorator_registers_at_decoration(self):
        @provide("custom.rpc.name")
        def get_country(lon: str, lat: str) -> str:
            return "IT"

        self.mock_instance.provide.assert_called_once_with("custom.rpc.name", get_country)

    def test_decorators_reject_methods(self):
        """Decorating a method or classmethod must fail at decoration time."""
        for decorator in (notify(), call(), provide()):
            with self.assertRaises(TypeError):

                @decorator
                def fake_method(self, value): ...


if __name__ == "__main__":
    unittest.main()
