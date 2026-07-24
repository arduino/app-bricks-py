# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

import pytest

from arduino.app_peripherals.camera.errors import CameraOpenError
from arduino.app_peripherals.camera.utils import claim_nth_available_camera


@pytest.fixture
def plugged_cameras(monkeypatch):
    """Declare how many USB and CSI cameras are plugged."""

    def configure(usb=0, csi=0):
        monkeypatch.setattr(
            "arduino.app_peripherals.camera.v4l_camera.V4LCamera.list_devices",
            staticmethod(lambda: list(range(usb))),
        )
        monkeypatch.setattr(
            "arduino.app_peripherals.camera.csi_camera.CSICamera.list_devices",
            staticmethod(lambda: list(range(csi))),
        )

    return configure


class TestClaimNthAvailableCamera:
    """Claim-aware device resolution used by Camera auto-selection."""

    def test_usb_takes_precedence_over_csi(self, plugged_cameras):
        plugged_cameras(usb=1, csi=1)

        assert claim_nth_available_camera(0) == "usb:0"

    def test_skips_already_claimed_cameras(self, plugged_cameras):
        plugged_cameras(usb=2)

        assert claim_nth_available_camera(0) == "usb:0"
        assert claim_nth_available_camera(0) == "usb:1"

    def test_falls_back_to_csi_when_usb_cameras_are_claimed(self, plugged_cameras):
        plugged_cameras(usb=1, csi=1)

        assert claim_nth_available_camera(0) == "usb:0"
        assert claim_nth_available_camera(0) == "csi:0"

    def test_index_counts_across_usb_and_csi(self, plugged_cameras):
        plugged_cameras(usb=1, csi=2)

        assert claim_nth_available_camera(1) == "csi:0"
        assert claim_nth_available_camera(2) == "csi:1"

    def test_reuses_plugged_order_when_all_claimed(self, plugged_cameras):
        plugged_cameras(usb=1)

        assert claim_nth_available_camera(0) == "usb:0"
        assert claim_nth_available_camera(0) == "usb:0"

    def test_csi_is_not_probed_when_usb_satisfies_the_index(self, plugged_cameras, monkeypatch):
        plugged_cameras(usb=1)
        probed = []
        monkeypatch.setattr(
            "arduino.app_peripherals.camera.csi_camera.CSICamera.list_devices",
            staticmethod(lambda: probed.append(True) or []),
        )

        assert claim_nth_available_camera(0) == "usb:0"
        assert probed == []

    def test_no_cameras_raises(self, plugged_cameras):
        plugged_cameras()

        with pytest.raises(CameraOpenError):
            claim_nth_available_camera(0)

    def test_out_of_range_index_raises(self, plugged_cameras):
        plugged_cameras(usb=1, csi=1)

        with pytest.raises(CameraOpenError):
            claim_nth_available_camera(2)
