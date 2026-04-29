# SPDX-FileCopyrightText: Copyright (C) ARDUINO SRL (http://www.arduino.cc)
#
# SPDX-License-Identifier: MPL-2.0

import ctypes
import fcntl
import glob
import os
import re

from .errors import CameraOpenError


def _iowr(type_char, nr, size):
    return (3 << 30) | (size << 16) | (ord(type_char) << 8) | nr


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


def get_media_device_info(path) -> MediaDeviceInfo:
    """Return MediaDeviceInfo for a /dev/mediaX node."""
    fd = os.open(path, os.O_RDONLY)
    try:
        info = MediaDeviceInfo()
        fcntl.ioctl(fd, MEDIA_IOC_DEVICE_INFO, info)
        return info
    finally:
        os.close(fd)


def find_camss_media_device(expected_driver="qcom-camss") -> str:
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
