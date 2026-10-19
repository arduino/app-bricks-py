# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

import os
from unittest.mock import MagicMock, patch

import pytest

from arduino.app_peripherals.camera import CameraOpenError, csi_camss_discovery


@pytest.fixture
def plugin_dir(tmp_path, monkeypatch):
    """Simulate a system GStreamer plugin directory and a clean setup state."""
    directory = tmp_path / "gstreamer-1.0"
    directory.mkdir()
    monkeypatch.setattr(csi_camss_discovery, "_GST_PLUGIN_DIR_GLOBS", (str(directory),))
    monkeypatch.setattr(csi_camss_discovery, "_setup_done", False)
    monkeypatch.delenv("GST_PLUGIN_SYSTEM_PATH_1_0", raising=False)
    monkeypatch.delenv("GST_PLUGIN_SYSTEM_PATH", raising=False)
    return directory


def test_setup_gstreamer_excludes_camx_only_plugins(plugin_dir):
    (plugin_dir / "libgstqtiqmmfsrc.so").touch()
    (plugin_dir / "libgstlibcamera.so").touch()
    (plugin_dir / "libgstvideoconvertscale.so").touch()

    csi_camss_discovery.setup_gstreamer()

    filtered_dir = os.environ["GST_PLUGIN_SYSTEM_PATH_1_0"]
    exposed = sorted(os.listdir(filtered_dir))
    assert exposed == ["libgstlibcamera.so", "libgstvideoconvertscale.so"]
    assert os.path.realpath(os.path.join(filtered_dir, "libgstlibcamera.so")) == str(plugin_dir / "libgstlibcamera.so")


def test_setup_gstreamer_leaves_environment_untouched_without_camx_plugins(plugin_dir):
    (plugin_dir / "libgstlibcamera.so").touch()

    csi_camss_discovery.setup_gstreamer()

    assert "GST_PLUGIN_SYSTEM_PATH_1_0" not in os.environ


def test_setup_gstreamer_is_idempotent(plugin_dir):
    (plugin_dir / "libgstqtiqmmfsrc.so").touch()
    (plugin_dir / "libgstlibcamera.so").touch()

    csi_camss_discovery.setup_gstreamer()
    filtered_dir = os.environ["GST_PLUGIN_SYSTEM_PATH_1_0"]
    csi_camss_discovery.setup_gstreamer()

    assert os.environ["GST_PLUGIN_SYSTEM_PATH_1_0"] == filtered_dir


def test_setup_gstreamer_honors_existing_plugin_path_override(plugin_dir, monkeypatch):
    (plugin_dir / "libgstqtiqmmfsrc.so").touch()
    (plugin_dir / "libgstlibcamera.so").touch()
    monkeypatch.setattr(csi_camss_discovery, "_GST_PLUGIN_DIR_GLOBS", ("/nonexistent",))
    monkeypatch.setenv("GST_PLUGIN_SYSTEM_PATH_1_0", str(plugin_dir))

    csi_camss_discovery.setup_gstreamer()

    filtered_dir = os.environ["GST_PLUGIN_SYSTEM_PATH_1_0"]
    assert filtered_dir != str(plugin_dir)
    assert sorted(os.listdir(filtered_dir)) == ["libgstlibcamera.so"]


_DEVICE_MONITOR_OUTPUT = """Device found:

        name  : imx219 2-0010
        class : Video/Source
"""


def test_resolve_camera_name_matches_the_i2c_address():
    """The camera name is the device monitor entry carrying the sensor I2C address."""
    with patch.object(csi_camss_discovery.subprocess, "run", return_value=MagicMock(stdout=_DEVICE_MONITOR_OUTPUT)):
        assert csi_camss_discovery.resolve_camera_name("2-0010") == "imx219 2-0010"


def test_resolve_camera_name_raises_without_a_match():
    """An I2C address no device monitor entry carries is not a camera."""
    with patch.object(csi_camss_discovery.subprocess, "run", return_value=MagicMock(stdout=_DEVICE_MONITOR_OUTPUT)):
        with pytest.raises(CameraOpenError, match="No camera matches I2C address"):
            csi_camss_discovery.resolve_camera_name("3-0010")
