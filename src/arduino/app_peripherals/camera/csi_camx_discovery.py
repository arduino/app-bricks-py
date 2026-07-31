# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

import ctypes
import os
import re
import stat

_PLATFORM_DRIVERS = "/sys/bus/platform/drivers"
_CAM_SERVER_SOCKET = "/run/cam_server/le_cam_socket"


def camx_driver_present() -> bool:
    """True if the downstream CamX/CSL platform drivers are bound on the host."""
    try:
        drivers = os.listdir(_PLATFORM_DRIVERS)
    except OSError:
        return False
    return any(d.startswith("cam_") for d in drivers)


def camx_socket_available() -> bool:
    """True if a cam-server socket is present (bind-mounted into the container)."""
    try:
        return stat.S_ISSOCK(os.stat(_CAM_SERVER_SOCKET).st_mode)
    except OSError:
        return False


# GStreamer (gi/Gst) and libglib are only available on CamX-capable hosts, so they're
# loaded lazily here rather than at module import time.


def _gst():
    """Import and initialize GStreamer's Python bindings on first use."""
    import gi

    gi.require_version("Gst", "1.0")
    from gi.repository import Gst

    Gst.init(None)
    return Gst


def _glib_hash_table_keys():
    """Bind the libglib GHashTable functions needed to read qtiqmmfsrc's static-metas property.

    Direct ctypes access is required because PyGObject can't read an opaque GHashTable.
    """
    glib = ctypes.CDLL("libglib-2.0.so.0")
    glib.g_hash_table_get_keys.restype = ctypes.POINTER(_GList)
    glib.g_hash_table_get_keys.argtypes = [ctypes.c_void_p]
    glib.g_list_free.argtypes = [ctypes.POINTER(_GList)]
    return glib


class _GList(ctypes.Structure):
    pass


_GList._fields_ = [("data", ctypes.c_void_p), ("next", ctypes.POINTER(_GList)), ("prev", ctypes.POINTER(_GList))]


# CAMX BACKEND INTERFACE (used by CSICamera)


def setup_gstreamer() -> None:
    """No-op: on CamX hosts every shipped GStreamer plugin is usable as-is."""


def list_camera_ids() -> list[int]:
    """List available CamX camera ids using the qtiqmmfsrc GStreamer plugin."""
    Gst = _gst()
    glib = _glib_hash_table_keys()

    # qtiqmmfsrc is only used to read the static-metas property and is never
    # started so no explicit teardown is needed.
    src = Gst.ElementFactory.make("qtiqmmfsrc", None)
    if src is None:
        raise SystemExit("qtiqmmfsrc plugin not available")

    metas = src.get_property("static-metas")  # GHashTable {camera_id (uint) -> CameraMetadata*}
    if metas is None:
        return []

    m = re.search(r"GHashTable at (0x[0-9a-fA-F]+)", repr(metas))
    ptr = int(m.group(1), 16) if m else None
    if not ptr:
        return []

    head = glib.g_hash_table_get_keys(ctypes.c_void_p(ptr))
    ids = []
    node = head
    while node:
        ids.append(node.contents.data or 0)
        node = node.contents.next
    glib.g_list_free(head)

    return sorted(int(x) for x in ids)


def get_camera_identifier(camera_id: int) -> str:
    """Return a synthetic camera name for the given CamX camera id (CamX has no real device name)."""
    return f"CAMERA{camera_id}"


def gstreamer_source(camera_id: int) -> str:
    """Build the qtiqmmfsrc GStreamer source element for the given CamX camera id."""
    return f"qtiqmmfsrc camera={camera_id}"
