# SPDX-FileCopyrightText: Copyright (C) ARDUINO SRL (http://www.arduino.cc)
#
# SPDX-License-Identifier: MPL-2.0

import pytest


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

    # Provide the parameter to tests so they can inject it into the constructor
    return request.param
