# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

import pytest

from arduino.app_peripherals.camera.errors import CameraOpenError
from arduino.app_peripherals.camera.utils import claim_first_available_camera, nth_plugged_camera


@pytest.fixture
def plugged_cameras(monkeypatch):
    """Declare how many USB and CSI cameras are plugged."""

    def configure(usb=0, csi=0):
        usb_paths = [f"/dev/v4l/by-id/usb-Cam{i}-video-index0" for i in range(usb)]
        csi_names = [f"CAMERA{i}" for i in range(csi)]
        monkeypatch.setattr(
            "arduino.app_peripherals.camera.v4l_camera.V4LCamera.list_devices",
            staticmethod(lambda: list(range(usb))),
        )
        monkeypatch.setattr(
            "arduino.app_peripherals.camera.v4l_camera.V4LCamera.list_stable_paths",
            staticmethod(lambda: usb_paths),
        )
        monkeypatch.setattr(
            "arduino.app_peripherals.camera.csi_camera.CSICamera.list_devices",
            staticmethod(lambda: list(range(csi))),
        )
        monkeypatch.setattr(
            "arduino.app_peripherals.camera.csi_camera.CSICamera.list_device_names",
            staticmethod(lambda: csi_names),
        )
        return usb_paths, csi_names

    return configure


class TestClaimFirstAvailableCamera:
    """Claim-aware device resolution used by Camera auto-selection."""

    def test_usb_takes_precedence_over_csi(self, plugged_cameras):
        usb_paths, _ = plugged_cameras(usb=1, csi=1)

        assert claim_first_available_camera() == (f"usb:{usb_paths[0]}", usb_paths[0])

    def test_skips_already_claimed_cameras(self, plugged_cameras):
        usb_paths, _ = plugged_cameras(usb=2)

        claim_first_available_camera()
        assert claim_first_available_camera() == (f"usb:{usb_paths[1]}", usb_paths[1])

    def test_falls_back_to_csi_when_usb_cameras_are_claimed(self, plugged_cameras):
        _, csi_names = plugged_cameras(usb=1, csi=1)

        claim_first_available_camera()
        assert claim_first_available_camera() == ("csi:0", csi_names[0])

    def test_raises_when_all_cameras_are_claimed(self, plugged_cameras):
        plugged_cameras(usb=1)

        claim_first_available_camera()
        with pytest.raises(CameraOpenError):
            claim_first_available_camera()

    def test_raises_when_no_camera_is_plugged(self, plugged_cameras):
        plugged_cameras()

        with pytest.raises(CameraOpenError):
            claim_first_available_camera()

    def test_csi_is_not_probed_when_a_usb_camera_is_available(self, plugged_cameras, monkeypatch):
        plugged_cameras(usb=1)
        probed = []
        monkeypatch.setattr(
            "arduino.app_peripherals.camera.csi_camera.CSICamera.list_device_names",
            staticmethod(lambda: probed.append(True) or []),
        )

        claim_first_available_camera()
        assert probed == []


class TestNthPluggedCamera:
    """Positional device resolution used by explicit selection, unaware of claims."""

    def test_usb_takes_precedence_over_csi(self, plugged_cameras):
        plugged_cameras(usb=1, csi=1)

        assert nth_plugged_camera(0) == "usb:0"

    def test_index_spans_usb_cameras_first_then_csi(self, plugged_cameras):
        plugged_cameras(usb=1, csi=2)

        assert nth_plugged_camera(1) == "csi:0"
        assert nth_plugged_camera(2) == "csi:1"

    def test_ignores_claims(self, plugged_cameras):
        plugged_cameras(usb=1)

        claim_first_available_camera()
        assert nth_plugged_camera(0) == "usb:0"

    def test_out_of_range_index_raises(self, plugged_cameras):
        plugged_cameras(usb=1)

        with pytest.raises(CameraOpenError):
            nth_plugged_camera(1)

    def test_no_cameras_raises(self, plugged_cameras):
        plugged_cameras()

        with pytest.raises(CameraOpenError):
            nth_plugged_camera(0)
