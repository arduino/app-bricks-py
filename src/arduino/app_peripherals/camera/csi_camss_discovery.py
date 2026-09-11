# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

import ctypes
import fcntl
import glob
import os
import re
import tempfile
import threading

from .errors import CameraOpenError
from .utils import resolve_camera_name


def _iowr(type_char: str, nr: int, size: int) -> int:
    return (3 << 30) | (size << 16) | (ord(type_char) << 8) | nr


_PLATFORM_DRIVERS = "/sys/bus/platform/drivers"


def camss_driver_present() -> bool:
    """True if the mainline qcom-camss platform driver is bound on the host."""
    return os.path.isdir(os.path.join(_PLATFORM_DRIVERS, "qcom-camss"))


# GSTREAMER PLUGIN FILTERING
#
# The qtiqmmfsrc plugin aborts at load time when cam-server is absent, crashing
# GStreamer's plugin scanner and leaving a core dump on every registry rebuild.
# On CAMSS hosts it can never work, so GStreamer is pointed to a filtered
# view of the plugin directories that omits it.

_CAMX_ONLY_PLUGINS = ("libgstqtiqmmfsrc",)
_GST_PLUGIN_DIR_GLOBS = (
    "/usr/lib/gstreamer-1.0",
    "/usr/lib/*/gstreamer-1.0",
    "/usr/local/lib/gstreamer-1.0",
    "/usr/local/lib/*/gstreamer-1.0",
)

_setup_lock = threading.Lock()
_setup_done = False


def _is_camx_only(filename: str) -> bool:
    return any(filename.startswith(name) for name in _CAMX_ONLY_PLUGINS)


def _gst_plugin_dirs() -> list[str]:
    """Return the directories GStreamer scans for plugins."""
    env_path = os.environ.get("GST_PLUGIN_SYSTEM_PATH_1_0") or os.environ.get("GST_PLUGIN_SYSTEM_PATH")
    if env_path:
        return [d for d in env_path.split(os.pathsep) if d]
    dirs = []
    for pattern in _GST_PLUGIN_DIR_GLOBS:
        dirs.extend(sorted(glob.glob(pattern)))
    return dirs


def setup_gstreamer() -> None:
    """
    Exclude CamX-only plugins from GStreamer's plugin search path.

    Idempotent and best-effort: if no CamX-only plugin is installed, the
    environment is left untouched.
    """
    global _setup_done
    with _setup_lock:
        if _setup_done:
            return
        _setup_done = True

        plugins = {}  # filename -> full path, first directory wins as in GStreamer
        for directory in _gst_plugin_dirs():
            try:
                entries = sorted(os.listdir(directory))
            except OSError:
                continue
            for entry in entries:
                if entry.endswith(".so"):
                    plugins.setdefault(entry, os.path.join(directory, entry))

        if not any(_is_camx_only(name) for name in plugins):
            return

        filtered_dir = tempfile.mkdtemp(prefix="camss-gst-plugins-")
        for name, path in plugins.items():
            if not _is_camx_only(name):
                os.symlink(path, os.path.join(filtered_dir, name))
        os.environ["GST_PLUGIN_SYSTEM_PATH_1_0"] = filtered_dir


# QCOM-CAMSS MEDIA DEVICE DISCOVERY


class MediaDeviceInfo(ctypes.Structure):
    _fields_ = [
        ("driver", ctypes.c_char * 16),
        ("model", ctypes.c_char * 32),
        ("serial", ctypes.c_char * 40),
        ("bus_info", ctypes.c_char * 32),
        ("media_version", ctypes.c_uint32),
        ("hw_revision", ctypes.c_uint32),
        ("driver_version", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32 * 31),
    ]


MEDIA_IOC_DEVICE_INFO = _iowr("|", 0x00, ctypes.sizeof(MediaDeviceInfo))


def get_media_device_info(path: str) -> MediaDeviceInfo:
    """Return MediaDeviceInfo for a /dev/mediaX node."""
    fd = os.open(path, os.O_RDONLY)
    try:
        info = MediaDeviceInfo()
        fcntl.ioctl(fd, MEDIA_IOC_DEVICE_INFO, info)
        return info
    finally:
        os.close(fd)


def find_camss_media_device(expected_driver: str = "qcom-camss") -> str:
    """Return the media device driven by qcom-camss."""
    for path in sorted(glob.glob("/dev/media*")):
        try:
            info = get_media_device_info(path)
            if info.driver.decode() == expected_driver:
                return path
        except OSError:
            continue

    raise RuntimeError(f"No media device found with driver '{expected_driver}'")


# QCOM-CAMSS MEDIA DEVICE'S MEDIA GRAPH PARSING


class MediaEntityInfo(ctypes.Union):
    _fields_ = [("raw", ctypes.c_uint8 * 184)]


class MediaEntityDesc(ctypes.Structure):
    _fields_ = [
        ("id", ctypes.c_uint32),
        ("name", ctypes.c_char * 32),
        ("type", ctypes.c_uint32),
        ("revision", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("group_id", ctypes.c_uint32),
        ("pads", ctypes.c_uint16),
        ("links", ctypes.c_uint16),
        ("reserved", ctypes.c_uint32 * 4),
        ("info", MediaEntityInfo),
    ]


class MediaPadDesc(ctypes.Structure):
    _fields_ = [
        ("entity", ctypes.c_uint32),
        ("index", ctypes.c_uint16),
        ("flags", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32 * 2),
    ]


class MediaLinkDesc(ctypes.Structure):
    _fields_ = [
        ("source", MediaPadDesc),
        ("sink", MediaPadDesc),
        ("flags", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32 * 2),
    ]


class MediaLinksEnum(ctypes.Structure):
    _fields_ = [
        ("entity", ctypes.c_uint32),
        ("pads", ctypes.POINTER(MediaPadDesc)),
        ("links", ctypes.POINTER(MediaLinkDesc)),
        ("reserved", ctypes.c_uint32 * 4),
    ]


MEDIA_IOC_ENUM_ENTITIES = _iowr("|", 0x01, ctypes.sizeof(MediaEntityDesc))
MEDIA_IOC_ENUM_LINKS = _iowr("|", 0x02, ctypes.sizeof(MediaLinksEnum))

MEDIA_ENT_ID_FLAG_NEXT = 1 << 31
MEDIA_ENT_F_CAM_SENSOR = 0x20001
MEDIA_LNK_FL_IMMUTABLE = 1 << 1


def scan_sensor_i2c_addresses(media_dev: str) -> list[tuple[str, str]]:
    """
    Scan the media graph to find all sensors and their I2C addresses.
    Return a list of tuples (csiphy_name, i2c_address).
    """
    fd = os.open(media_dev, os.O_RDWR)
    sensors_found = []
    try:
        # Enumerate all entities
        entities = []
        desc = MediaEntityDesc()
        desc.id = MEDIA_ENT_ID_FLAG_NEXT
        while True:
            try:
                fcntl.ioctl(fd, MEDIA_IOC_ENUM_ENTITIES, desc)
            except OSError:
                break
            entities.append({
                "id": desc.id,
                "name": desc.name.decode(errors="ignore").rstrip("\x00"),
                "type": desc.type,
                "num_pads": desc.pads,
                "num_links": desc.links,
            })
            desc.id |= MEDIA_ENT_ID_FLAG_NEXT

        by_id = {e["id"]: e for e in entities}

        # For each sensor, look for the IMMUTABLE link to its CSIPHY
        for entity in entities:
            if entity["type"] != MEDIA_ENT_F_CAM_SENSOR:
                continue
            if entity["num_links"] == 0:
                continue

            pads = (MediaPadDesc * entity["num_pads"])()
            links = (MediaLinkDesc * entity["num_links"])()
            req = MediaLinksEnum()
            req.entity = entity["id"]
            req.pads = pads
            req.links = links
            fcntl.ioctl(fd, MEDIA_IOC_ENUM_LINKS, req)

            for i in range(entity["num_links"]):
                if not (links[i].flags & MEDIA_LNK_FL_IMMUTABLE):
                    continue
                sink = by_id.get(links[i].sink.entity)
                if sink and "msm_csiphy" in sink["name"]:
                    m = re.search(r"(\d+-[\da-fA-F]{4})", entity["name"])
                    if m:
                        sensors_found.append((sink["name"], m.group(1)))

        return sensors_found
    finally:
        os.close(fd)


def find_sensor_i2c_addr(media_dev: str, csiphy_index: int) -> str:
    """
    Traverse the media graph to find a sensor with an immutable link to the
    specified CSIPHY index.
    Return the I2C address of the found sensor.
    """
    csiphy_name = f"msm_csiphy{csiphy_index}"
    try:
        entities = scan_sensor_i2c_addresses(media_dev)
        for name, i2c_addr in entities:
            if name == csiphy_name:
                return i2c_addr

    except Exception as e:
        raise RuntimeError(f"Error scanning media graph: {e}")

    raise CameraOpenError(f"No sensor found on {csiphy_name}")


# CAMSS BACKEND INTERFACE (used by CSICamera)


def list_camera_ids() -> list[int]:
    """Return the sorted list of CSIPHY indices with a sensor attached."""
    media_dev = find_camss_media_device()
    ids = set()
    for csiphy_name, _ in scan_sensor_i2c_addresses(media_dev):
        m = re.search(r"msm_csiphy(\d+)", csiphy_name)
        if m:
            ids.add(int(m.group(1)))
    return sorted(ids)


def get_camera_identifier(camera_id: int) -> str:
    """Return the resolved sensor name for the sensor wired at the given CSIPHY index."""
    media_dev = find_camss_media_device()
    i2c_addr = find_sensor_i2c_addr(media_dev, camera_id)
    return resolve_camera_name(i2c_addr)


def gstreamer_source(camera_id: int) -> str:
    """Build the libcamerasrc GStreamer source element for the given CSIPHY index."""
    camera_name = get_camera_identifier(camera_id).replace(" ", r"\ ")  # Escape spaces for GStreamer pipeline
    return f"libcamerasrc camera-name={camera_name}"
