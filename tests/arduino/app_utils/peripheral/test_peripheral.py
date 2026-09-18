# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

# Validates the @peripheral class decorator: every instance of a decorated class is registered
# for release when the application shuts down, the same way @brick registers instances with the
# AppController.
import gc
import threading

import pytest

import arduino.app_utils.app as app
from arduino.app_utils import AppController, brick, peripheral, peripheral_registry
from arduino.app_utils.peripheral_registry import PeripheralRegistry


@pytest.fixture
def peripherals(monkeypatch):
    """Provides a fresh registry that the decorator registers into."""
    registry = PeripheralRegistry()
    monkeypatch.setattr(peripheral_registry, "Peripherals", registry)
    return registry


class TestDecorator:
    def test_instance_is_released_by_the_registry(self, peripherals):
        @peripheral
        class Device:
            def __init__(self):
                self.stopped = False

            def stop(self):
                self.stopped = True

        device = Device()
        peripherals.stop_all(timeout=1.0)

        assert device.stopped

    def test_can_be_used_with_parentheses(self, peripherals):
        @peripheral()
        class Device:
            def __init__(self):
                self.stopped = False

            def stop(self):
                self.stopped = True

        device = Device()
        peripherals.stop_all(timeout=1.0)

        assert device.stopped

    def test_preserves_init_signature_and_arguments(self, peripherals):
        @peripheral
        class Device:
            def __init__(self, name, *, rate=10):
                self.name = name
                self.rate = rate

            def stop(self):
                pass

        device = Device("cam", rate=30)

        assert (device.name, device.rate) == ("cam", 30)
        assert Device.__init__.__name__ == "__init__"

    def test_subclass_of_a_decorated_base_is_registered(self, peripherals):
        """Decorating the base class is enough: subclasses register through super().__init__()."""

        @peripheral
        class Base:
            def __init__(self):
                self.stopped = False

            def stop(self):
                self.stopped = True

        class Concrete(Base):
            def __init__(self, extra):
                super().__init__()
                self.extra = extra

        device = Concrete(extra=1)
        peripherals.stop_all(timeout=1.0)

        assert device.stopped

    def test_decorating_base_and_subclass_releases_once(self, peripherals):
        @peripheral
        class Base:
            def __init__(self):
                self.stop_count = 0

            def stop(self):
                self.stop_count += 1

        @peripheral
        class Concrete(Base):
            pass

        device = Concrete()
        peripherals.stop_all(timeout=1.0)

        assert device.stop_count == 1

    def test_failed_construction_is_not_registered(self, peripherals):
        """An instance whose __init__ raised must never be asked to stop."""
        stops = []

        @peripheral
        class Device:
            def __init__(self):
                raise ValueError("bad config")

            def stop(self):
                stops.append(self)

        with pytest.raises(ValueError):
            Device()
        peripherals.stop_all(timeout=1.0)

        assert stops == []

    def test_does_not_keep_instances_alive(self, peripherals):
        @peripheral
        class Device:
            def stop(self):
                pass

        device = Device()
        del device
        gc.collect()

        assert peripherals.stop_all(timeout=1.0) == []


class TestWithApp:
    @pytest.fixture
    def app_instance(self, monkeypatch):
        instance = AppController()
        monkeypatch.setattr(app, "App", instance)
        return instance

    @pytest.fixture(autouse=True)
    def fast_budget(self, monkeypatch):
        monkeypatch.setattr(app, "SHUTDOWN_BRICKS_BUDGET_S", 0.6)
        monkeypatch.setattr(app, "SHUTDOWN_PERIPHERALS_BUDGET_S", 0.2)
        monkeypatch.setattr(app, "SHUTDOWN_LOCK_BUDGET_S", 0.1)

    def test_user_defined_peripheral_is_released_after_the_bricks(self, app_instance, peripherals):
        """The public contract: a user class decorated with @peripheral is released on shutdown,
        after every brick, without the user calling stop() themselves.
        """
        timeline: list[str] = []

        @brick
        class Worker:
            def stop(self):
                timeline.append("brick")

        @peripheral
        class Sensor:
            def stop(self):
                timeline.append("peripheral")

        Worker()
        sensor = Sensor()
        assert sensor is not None

        thread = threading.Thread(target=app_instance.run, daemon=True)
        thread.start()
        deadline = threading.Event()
        deadline.wait(0.2)
        app_instance._shutdown()

        assert timeline == ["brick", "peripheral"]
