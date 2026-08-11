# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

import gc

import pytest

from arduino.app_peripherals.device_registry import DeviceRegistry


@pytest.fixture
def registry():
    return DeviceRegistry()


class TestSelect:
    """Auto-selection: claim the first available device."""

    def test_selects_first_listed_device(self, registry):
        assert registry.select(lambda: ["a", "b"]) == "a"

    def test_skips_already_claimed_devices(self, registry):
        registry.select(lambda: ["a", "b"])

        assert registry.select(lambda: ["a", "b"]) == "b"

    def test_returns_none_when_all_devices_are_claimed(self, registry):
        registry.select(lambda: ["a"])

        assert registry.select(lambda: ["a"]) is None

    def test_returns_none_when_no_device_is_listed(self, registry):
        assert registry.select(lambda: []) is None

    def test_groups_are_enumerated_in_precedence_order(self, registry):
        registry.select(lambda: ["a"], lambda: ["b"])

        assert registry.select(lambda: ["a"], lambda: ["b"]) == "b"

    def test_later_groups_are_not_enumerated_when_a_device_is_available(self, registry):
        probed = []

        def second_group():
            probed.append(True)
            return ["b"]

        assert registry.select(lambda: ["a"], second_group) == "a"
        assert probed == []

    def test_skips_devices_claimed_explicitly(self, registry):
        registry.claim("a")

        assert registry.select(lambda: ["a", "b"]) == "b"


class TestClaim:
    """Explicit claims: unconditional and counted."""

    def test_claiming_an_already_claimed_device_is_allowed(self, registry):
        registry.claim("a")
        registry.claim("a")

        assert registry.select(lambda: ["a", "b"]) == "b"

    def test_shared_device_stays_claimed_until_all_owners_release_it(self, registry):
        registry.claim("a")
        registry.claim("a")

        registry.release("a")
        assert registry.select(lambda: ["a", "b"]) == "b"

        registry.release("a")
        assert registry.select(lambda: ["a", "b"]) == "a"


class TestRelease:
    """Claims lifecycle: release and owner binding."""

    def test_release_makes_device_available_again(self, registry):
        registry.select(lambda: ["a"])
        registry.release("a")

        assert registry.select(lambda: ["a", "b"]) == "a"

    def test_release_of_unclaimed_device_is_a_noop(self, registry):
        registry.release("a")

        assert registry.select(lambda: ["a"]) == "a"

    def test_bound_claim_is_released_when_owner_is_garbage_collected(self, registry):
        class Owner:
            pass

        owner = Owner()
        registry.select(lambda: ["a"])
        registry.bind("a", owner)

        del owner
        gc.collect()

        assert registry.select(lambda: ["a", "b"]) == "a"

    def test_clear_drops_all_claims(self, registry):
        registry.select(lambda: ["a"])
        registry.claim("b")
        registry.clear()

        assert registry.select(lambda: ["a", "b"]) == "a"
