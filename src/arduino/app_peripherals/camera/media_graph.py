# SPDX-FileCopyrightText: Copyright (C) ARDUINO SRL (http://www.arduino.cc)
#
# SPDX-License-Identifier: MPL-2.0

import ctypes
import fcntl
import os
import re
import subprocess
from .errors import CameraOpenError

MEDIA_ENT_ID_FLAG_NEXT = 1 << 31
MEDIA_ENT_F_CAM_SENSOR = 0x20001
MEDIA_LNK_FL_IMMUTABLE = 1 << 1


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


def _iowr(type_char, nr, size):
    return (3 << 30) | (size << 16) | (ord(type_char) << 8) | nr


MEDIA_IOC_ENUM_ENTITIES = _iowr("|", 0x01, ctypes.sizeof(MediaEntityDesc))
MEDIA_IOC_ENUM_LINKS = _iowr("|", 0x02, ctypes.sizeof(MediaLinksEnum))


def scan_sensor_i2c_addresses(media_dev):
    """
    Scan the media graph to find all sensors and their I2C addresses.
    Return a list of tuples (sensor_name, i2c_address).
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


def find_sensor_i2c_addr(media_dev, csiphy_index):
    """
    Traverse the media graph to find CSIPHY a sensor with an immutable link to the provided CSIPHY.
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


def resolve_camera_name(i2c_addr) -> str:
    """
    Find the camera name corresponding to the given I2C address.
    """
    output = subprocess.run(
        ["gst-device-monitor-1.0", "Video/Source"],
        capture_output=True,
        text=True,
        timeout=10,
    ).stdout

    for line in output.splitlines():
        m = re.match(r"^\s+name\s+:\s+(.+)$", line)
        if m and i2c_addr in m.group(1):
            return m.group(1).strip()

    raise CameraOpenError(f"No camera matches I2C address '{i2c_addr}'")
