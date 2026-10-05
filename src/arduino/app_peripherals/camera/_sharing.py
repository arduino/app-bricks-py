# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""Sharing of a camera device between several handles of the same process.

A local camera can be streamed by a single client at a time: a second V4L handle fails with
EBUSY and the CSI camera service assigns a camera to one client only. The Camera factory hands
out a follower handle when a device is already held by another instance, the leader, and this
module fans the leader's frames out to every handle, so that each one receives every frame.

The device is read on demand by whichever handle first asks for a new frame, so the frame rate
is the one of the leader and no extra thread is used. Only the latest frame is kept: a handle
slower than the camera skips the frames it missed, and a frame older than one frame interval is
never handed out, a new one is read instead.
"""

import threading
import time
import weakref
from collections.abc import Callable
from typing import Any, Literal

import numpy as np

from .base_camera import BaseCamera
from .errors import CameraReadError


class FrameHub:
    """Fans the frames of a leader camera out to the follower handles sharing its device."""

    def __init__(self, leader: BaseCamera) -> None:
        """
        Initialize the hub.

        Args:
            leader (BaseCamera): The camera that owns the device.
        """
        self.leader = leader
        self._followers: weakref.WeakSet[BaseCamera] = weakref.WeakSet()
        # Guards the followers on its own: status changes are notified from within start() and
        # stop(), while the lifecycle lock is held
        self._followers_lock = threading.Lock()

        # Held across a whole start() or stop(), so that the device is never closed while a handle
        # is being started. Always acquired before the leader's camera lock.
        self._lifecycle_lock = threading.Lock()

        # Never held across a device read, which blocks for up to a frame interval
        self._frame_cond = threading.Condition()
        self._frame: np.ndarray | None = None
        self._frame_seq = 0
        self._frame_time = 0.0
        self._producing = False

    def add_follower(self, follower: BaseCamera) -> None:
        """Register a follower handle."""
        with self._followers_lock:
            self._followers.add(follower)

    def _handles(self) -> list[BaseCamera]:
        """The leader and the followers still alive."""
        with self._followers_lock:
            return [self.leader, *self._followers]

    def start(self, handle: BaseCamera) -> None:
        """Start a handle, opening the device if no other handle has already done it."""
        with self._lifecycle_lock:
            handle._user_started = True  # pyright: ignore[reportPrivateUsage]
            try:
                self.leader._start_device()  # pyright: ignore[reportPrivateUsage]
            except BaseException:
                handle._user_started = False  # pyright: ignore[reportPrivateUsage]
                raise

    def stop(self, handle: BaseCamera) -> None:
        """Stop a handle, closing the device if no handle is left started."""
        with self._lifecycle_lock:
            handle._user_started = False  # pyright: ignore[reportPrivateUsage]
            # Evaluated whatever the state of this handle, so that stopping any handle releases a
            # device left open by a started handle that was garbage collected
            if any(h._user_started for h in self._handles()):  # pyright: ignore[reportPrivateUsage]
                return
            self.leader._stop_device()  # pyright: ignore[reportPrivateUsage]

        # Wake up the handles waiting for a frame, so they notice the device is closed
        with self._frame_cond:
            self._frame = None
            self._frame_cond.notify_all()

    def capture(self, handle: BaseCamera) -> np.ndarray | None:
        """
        Capture the next frame for a handle.

        Returns the latest frame read from the device if the handle has not received it yet and it
        is not older than one frame interval. Otherwise waits for the next frame, reading it from
        the device if no other handle is already doing it.

        Args:
            handle (BaseCamera): The handle asking for a frame.

        Returns:
            The frame with the handle's adjustments applied, or None if no frame is available.
            The frame read from the device is read-only, as it is shared with the other handles.

        Raises:
            CameraReadError: If the handle is not started.
            CameraTransformError: If the handle's adjustments fail.
            Exception: If the device fails to read a frame.
        """
        latest = None
        with self._frame_cond:
            while True:
                if not handle.is_started():
                    raise CameraReadError(f"Attempted to read from {handle.name} before starting it.")

                if self._frame is not None and handle._shared_seq < self._frame_seq and self._is_fresh():  # pyright: ignore[reportPrivateUsage]
                    handle._shared_seq = self._frame_seq  # pyright: ignore[reportPrivateUsage]
                    latest = self._frame
                    break
                if not self._producing:
                    self._producing = True
                    break
                self._frame_cond.wait()

        # Adjustments are applied without holding the condition, not to hold up the other handles
        if latest is not None:
            return handle._apply_adjustments(latest)  # pyright: ignore[reportPrivateUsage]

        frame = None
        try:
            frame = self.leader._capture_device(adjust=False)  # pyright: ignore[reportPrivateUsage]
        finally:
            with self._frame_cond:
                self._producing = False
                if frame is not None:
                    frame.flags.writeable = False
                    self._frame = frame
                    self._frame_seq += 1
                    self._frame_time = time.monotonic()
                    handle._shared_seq = self._frame_seq  # pyright: ignore[reportPrivateUsage]
                self._frame_cond.notify_all()

        return handle._apply_adjustments(frame) if frame is not None else None  # pyright: ignore[reportPrivateUsage]

    def notify_status(self, status: str, data: dict[str, Any]) -> None:
        """Forward a status change of the device to the status callbacks of the followers."""
        with self._followers_lock:
            followers = list(self._followers)
        for follower in followers:
            callback = follower._on_status_changed_cb  # pyright: ignore[reportPrivateUsage]
            if callback is not None:
                follower._event_executor.submit(callback, status, data)  # pyright: ignore[reportPrivateUsage]

    def _is_fresh(self) -> bool:
        """Whether the latest frame is not older than one frame interval."""
        return time.monotonic() - self._frame_time < 1.0 / self.leader.fps


class CameraFollower(BaseCamera):
    """
    Handle on a camera device already held by another instance in this process.

    It behaves like the leader camera, whose resolution, FPS and settings it uses, except for the
    adjustments, which are its own. Attributes specific to the camera type, such as v4l_path or
    csi_path, are read from the leader.
    """

    def __init__(self, hub: FrameHub, adjustments: Callable[[np.ndarray], np.ndarray] | None) -> None:
        """
        Initialize the follower.

        Args:
            hub (FrameHub): The hub sharing the leader's device.
            adjustments (callable, optional): Function or function pipeline to adjust the frames of
                this handle only.
        """
        # Set ahead of the base initializer, which assigns resolution and fps through the
        # properties below
        self._leader = hub.leader
        super().__init__(resolution=hub.leader.resolution, fps=hub.leader.fps, adjustments=adjustments, auto_reconnect=False)
        self.name = hub.leader.name
        self.logger = hub.leader.logger

        # The follower keeps the hub alive, the leader only references it weakly
        self._hub = hub
        self._hub_ref = weakref.ref(hub)

    @property
    def leader(self) -> BaseCamera:
        """The camera holding the device."""
        return self._leader

    @property
    def resolution(self) -> tuple[int, int]:  # pyright: ignore[reportIncompatibleVariableOverride]
        """Resolution of the device, as negotiated by the leader once started."""
        return self._leader.resolution

    @resolution.setter
    def resolution(self, value: tuple[int, int]) -> None:
        self._leader.resolution = value

    @property
    def fps(self) -> int:
        """Frames per second of the device, as negotiated by the leader once started."""
        return self._leader.fps

    @fps.setter
    def fps(self, value: int) -> None:
        self._leader.fps = value

    @property
    def status(self) -> Literal["disconnected", "connected", "streaming", "paused"]:
        """Status of the device."""
        return self._leader.status

    def __getattr__(self, name: str) -> Any:  # noqa: ANN401
        """Read the attributes specific to the camera type, such as v4l_path, from the leader."""
        if name.startswith("_"):
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
        return getattr(self._leader, name)

    def _open_camera(self) -> None:
        raise RuntimeError("A camera follower never opens the device itself")

    def _close_camera(self) -> None:
        raise RuntimeError("A camera follower never closes the device itself")

    def _read_frame(self) -> np.ndarray | None:
        raise RuntimeError("A camera follower never reads the device itself")


_hubs_lock = threading.Lock()


def follow(leader: BaseCamera, adjustments: Callable[[np.ndarray], np.ndarray] | None) -> CameraFollower:
    """
    Create a new handle on the device held by a leader camera.

    Args:
        leader (BaseCamera): The camera holding the device.
        adjustments (callable, optional): Adjustments of the new handle only.

    Returns:
        CameraFollower: The new handle.
    """
    with _hubs_lock:
        hub = leader._get_hub()  # pyright: ignore[reportPrivateUsage]
        if hub is None:
            hub = FrameHub(leader)
            leader._hub_ref = weakref.ref(hub)  # pyright: ignore[reportPrivateUsage]
        follower = CameraFollower(hub, adjustments)
        hub.add_follower(follower)
    return follower
