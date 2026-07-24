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
    """Claim-aware selection of the n-th available device."""

    def test_selects_devices_in_plugged_order(self, registry):
        assert registry.select(0, lambda: ["a", "b"]) == "a"

    def test_skips_already_claimed_devices(self, registry):
        registry.select(0, lambda: ["a", "b"])

        assert registry.select(0, lambda: ["a", "b"]) == "b"

    def test_indexes_among_available_devices(self, registry):
        registry.select(0, lambda: ["a", "b", "c"])

        # "a" is claimed, so the second available device is "c".
        assert registry.select(1, lambda: ["a", "b", "c"]) == "c"

    def test_reuses_nth_plugged_device_when_pool_is_exhausted(self, registry):
        registry.select(0, lambda: ["a"])

        assert registry.select(0, lambda: ["a"]) == "a"

    def test_returns_none_when_no_device_at_index(self, registry):
        assert registry.select(2, lambda: ["a", "b"]) is None
        assert registry.select(-1, lambda: ["a", "b"]) is None
        assert registry.select(0, lambda: []) is None

    def test_groups_are_enumerated_in_precedence_order(self, registry):
        assert registry.select(1, lambda: ["a"], lambda: ["b"]) == "b"

    def test_later_groups_are_not_enumerated_when_index_is_satisfied(self, registry):
        probed = []

        def second_group():
            probed.append(True)
            return ["b"]

        assert registry.select(0, lambda: ["a"], second_group) == "a"
        assert probed == []


class TestRelease:
    """Claims lifecycle: release, shared claims and owner binding."""

    def test_release_makes_device_available_again(self, registry):
        registry.select(0, lambda: ["a"])
        registry.release("a")

        assert registry.select(0, lambda: ["a", "b"]) == "a"

    def test_release_of_unclaimed_device_is_a_noop(self, registry):
        registry.release("a")

        assert registry.select(0, lambda: ["a"]) == "a"

    def test_shared_device_stays_claimed_until_all_owners_release_it(self, registry):
        registry.select(0, lambda: ["a"])
        registry.select(0, lambda: ["a"])  # Pool exhausted: "a" is shared

        registry.release("a")
        assert registry.select(0, lambda: ["a", "b"]) == "b"

        registry.release("a")
        assert registry.select(0, lambda: ["a", "b"]) == "a"

    def test_bound_claim_is_released_when_owner_is_garbage_collected(self, registry):
        class Owner:
            pass

        owner = Owner()
        registry.select(0, lambda: ["a"])
        registry.bind("a", owner)

        del owner
        gc.collect()

        assert registry.select(0, lambda: ["a", "b"]) == "a"

    def test_clear_drops_all_claims(self, registry):
        registry.select(0, lambda: ["a"])
        registry.select(0, lambda: ["b"])
        registry.clear()

        assert registry.select(0, lambda: ["a", "b"]) == "a"
