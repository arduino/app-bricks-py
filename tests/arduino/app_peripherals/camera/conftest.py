# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

import pytest

from arduino.app_peripherals.camera.utils import _camera_registry


@pytest.fixture(autouse=True)
def clean_camera_registry():
    """Give each test a clean slate of auto-selected camera claims."""
    _camera_registry.clear()
    yield


def _fake_v4l_devices(monkeypatch, links, capture_targets=None):
    """
    Patch os functions in v4l_camera to simulate the given /dev/v4l/by-id links.

    Args:
        monkeypatch: The pytest monkeypatch fixture.
        links: Mapping of stable /dev/v4l/by-id paths to /dev/videoN targets.
        capture_targets: Targets supporting video capture (all of them if None).
    """
    by_id_dir = "/dev/v4l/by-id/"

    monkeypatch.setattr("arduino.app_peripherals.camera.v4l_camera.os.path.exists", lambda path: True)
    monkeypatch.setattr("arduino.app_peripherals.camera.v4l_camera.os.path.islink", lambda path: path in links)
    monkeypatch.setattr(
        "arduino.app_peripherals.camera.v4l_camera.os.listdir",
        lambda path: [entry.removeprefix(by_id_dir) for entry in links] if path == by_id_dir else [],
    )
    monkeypatch.setattr("arduino.app_peripherals.camera.v4l_camera.os.path.realpath", lambda path: links.get(path, path))
    monkeypatch.setattr(
        "arduino.app_peripherals.camera.v4l_camera.V4LCamera._supports_video_capture",
        staticmethod(lambda path: capture_targets is None or path in capture_targets),
    )


@pytest.fixture
def two_v4l_cameras(monkeypatch):
    """
    Simulate two stable USB cameras:
    /dev/v4l/by-id/usb-CamA-video-index0 -> /dev/video0
    /dev/v4l/by-id/usb-CamB-video-index0 -> /dev/video2
    """
    by_id_dir = "/dev/v4l/by-id/"
    _fake_v4l_devices(
        monkeypatch,
        {
            by_id_dir + "usb-CamA-video-index0": "/dev/video0",
            by_id_dir + "usb-CamB-video-index0": "/dev/video2",
        },
    )


@pytest.fixture
def usb_camera_with_metadata_node(monkeypatch):
    """
    Simulate a single UVC camera exposing a capture node and a metadata node:
    /dev/v4l/by-id/usb-Cam-video-index0 -> /dev/video10 (capture)
    /dev/v4l/by-id/usb-Cam-video-index1 -> /dev/video11 (metadata, not capture-capable)
    """
    by_id_dir = "/dev/v4l/by-id/"
    _fake_v4l_devices(
        monkeypatch,
        {
            by_id_dir + "usb-Cam-video-index0": "/dev/video10",
            by_id_dir + "usb-Cam-video-index1": "/dev/video11",
        },
        capture_targets={"/dev/video10"},
    )


class _FakeCSIBackend:
    """Stand-in for a CSI camera stack backend exposing two cameras."""

    @staticmethod
    def list_camera_ids():
        return [0, 1]

    @staticmethod
    def get_camera_identifier(camera_id):
        return f"CAMERA{camera_id}"

    @staticmethod
    def gstreamer_source(camera_id):
        return f"fakesrc camera={camera_id}"


@pytest.fixture
def two_csi_cameras_only(monkeypatch):
    """Simulate a platform with two CSI cameras (CAMERA0, CAMERA1) and no USB camera."""
    monkeypatch.setattr("arduino.app_peripherals.camera.v4l_camera.V4LCamera._scan_stable_links", staticmethod(lambda: []))
    monkeypatch.setattr("arduino.app_peripherals.camera.csi_camera._get_backend", lambda: _FakeCSIBackend)


@pytest.fixture(
    params=["/dev/video0", 0, "0", "/dev/v4l/by-path/platform-xhci-hcd.2.auto-usb-0:1.3:1.0-video-index0", "/dev/v4l/by-id/usb-Camera-video-index0"]
)
def v4l_device_argument(monkeypatch, request):
    """
    Patch os functions for V4LCamera stable path resolution to simulate a stable
    camera environment for various device arguments.
    The only valid resolved device is "/dev/v4l/by-id/usb-Camera-video-index0".
    """
    fake_by_id_dir = "/dev/v4l/by-id/"
    fake_by_id_entry = "usb-Camera-video-index0"
    fake_by_id_path = fake_by_id_dir + fake_by_id_entry
    fake_video_path = "/dev/video0"

    def fake_exists(path):
        # All relevant paths exist
        return True

    def fake_islink(path):
        # Only the fake by-id path is a symlink
        return path == fake_by_id_path

    def fake_listdir(path):
        # Only one entry in by-id
        if path == fake_by_id_dir:
            return [fake_by_id_entry]
        return []

    def fake_realpath(path):
        # The by-id symlink points to /dev/video0
        if path == fake_by_id_path:
            return fake_video_path
        # by-path resolves to /dev/video0
        if path.startswith("/dev/v4l/by-path"):
            return fake_video_path
        return path

    monkeypatch.setattr("arduino.app_peripherals.camera.v4l_camera.os.path.exists", fake_exists)
    monkeypatch.setattr("arduino.app_peripherals.camera.v4l_camera.os.path.islink", fake_islink)
    monkeypatch.setattr("arduino.app_peripherals.camera.v4l_camera.os.listdir", fake_listdir)
    monkeypatch.setattr("arduino.app_peripherals.camera.v4l_camera.os.path.realpath", fake_realpath)
    monkeypatch.setattr("arduino.app_peripherals.camera.v4l_camera.V4LCamera._supports_video_capture", staticmethod(lambda path: True))

    # Provide the parameter to tests so they can inject it into the constructor
    return request.param
