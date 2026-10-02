# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

import threading
import time

import numpy as np
import pytest

from arduino.app_peripherals.camera import BaseCamera, CameraConfigError, CameraReadError, SharedCamera
from arduino.app_utils.peripheral_registry import Peripherals


class CountingCamera(BaseCamera):
    """Camera returning a new frame per read, whose pixels hold the read number."""

    def __init__(self, fps: int = 20, fail_read: bool = False):
        super().__init__(resolution=(4, 2), fps=fps, auto_reconnect=False)
        self.fail_read = fail_read
        self.open_count = 0
        self.close_count = 0
        self.read_count = 0

    def _open_camera(self):
        self.open_count += 1
        self.resolution = (8, 6)  # Renegotiated by the device, like V4L and CSI cameras do

    def _close_camera(self):
        self.close_count += 1

    def _read_frame(self):
        if self.fail_read:
            raise RuntimeError("read failed")
        self.read_count += 1
        return np.full((6, 8, 3), self.read_count, dtype=np.int32)


def _frame_id(frame: np.ndarray) -> int:
    return int(frame[0, 0, 0])


def _run_in_threads(*targets, timeout: float = 5.0) -> None:
    threads = [threading.Thread(target=target, daemon=True) for target in targets]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout)
        assert not thread.is_alive(), "consumer thread hung"


def test_shared_camera_mirrors_source():
    source = CountingCamera(fps=15)
    camera = SharedCamera(source)

    assert camera.source is source
    assert camera.fps == 15
    assert camera.resolution == (4, 2)

    camera.start()
    try:
        assert camera.is_started()
        assert camera.resolution == (8, 6)  # Follows the resolution negotiated on start
        assert camera.status == source.status
    finally:
        camera.stop()


def test_shared_camera_rejects_invalid_source():
    with pytest.raises(CameraConfigError):
        SharedCamera(SharedCamera(CountingCamera()))
    with pytest.raises(CameraConfigError):
        SharedCamera("usb:0")  # type: ignore[arg-type]


def test_concurrent_consumers_receive_the_same_frames():
    """Consumers capturing together get the same frame objects, read once from the source."""
    source = CountingCamera()
    camera = SharedCamera(source)
    barrier = threading.Barrier(2)
    received: dict[str, list[np.ndarray]] = {"a": [], "b": []}

    def consumer(name: str):
        for _ in range(8):
            barrier.wait()
            frame = camera.capture()
            assert frame is not None
            received[name].append(frame)

    with camera:
        _run_in_threads(lambda: consumer("a"), lambda: consumer("b"))

    assert [_frame_id(f) for f in received["a"]] == list(range(1, 9))
    assert all(a is b for a, b in zip(received["a"], received["b"]))
    assert source.read_count == 8


def test_single_consumer_reads_every_frame_from_source():
    """With one consumer the shared camera behaves like the source: no extra or skipped reads."""
    source = CountingCamera(fps=50)
    camera = SharedCamera(source)

    with camera:
        ids = [_frame_id(f) for f in (camera.capture() for _ in range(5)) if f is not None]

    assert ids == [1, 2, 3, 4, 5]
    assert source.read_count == 5


def test_consumer_never_receives_a_frame_twice():
    source = CountingCamera(fps=50)
    camera = SharedCamera(source)
    received: dict[str, list[int]] = {"fast": [], "slow": []}

    def consumer(name: str, work: float):
        deadline = time.monotonic() + 0.6
        while time.monotonic() < deadline:
            frame = camera.capture()
            if frame is not None:
                received[name].append(_frame_id(frame))
            time.sleep(work)

    with camera:
        _run_in_threads(lambda: consumer("fast", 0), lambda: consumer("slow", 0.1))

    for ids in received.values():
        assert ids == sorted(set(ids))
    # The slow consumer skips the frames it was too slow to see instead of queuing them
    assert len(received["slow"]) < len(received["fast"])


def test_slow_consumer_gets_the_latest_frame_without_waiting():
    source = CountingCamera(fps=5)
    camera = SharedCamera(source)
    received: list[int] = []

    def fast():
        for _ in range(3):
            frame = camera.capture()
            assert frame is not None
            received.append(_frame_id(frame))

    with camera:
        _run_in_threads(fast)

        start = time.monotonic()
        frame = camera.capture()  # Fresh frame produced by the other consumer
        elapsed = time.monotonic() - start

    assert frame is not None
    assert _frame_id(frame) == received[-1]
    assert elapsed < 0.1  # Well below the 200 ms source frame interval
    assert source.read_count == 3


def test_stale_frame_is_not_handed_out():
    source = CountingCamera(fps=20)
    camera = SharedCamera(source)

    with camera:
        _run_in_threads(lambda: camera.capture())
        time.sleep(0.1)  # Two source frame intervals
        frame = camera.capture()

    assert frame is not None
    assert _frame_id(frame) == 2
    assert source.read_count == 2


def test_shared_frames_are_read_only():
    camera = SharedCamera(CountingCamera(fps=50))

    with camera:
        frame = camera.capture()

    assert frame is not None
    with pytest.raises(ValueError):
        frame[0, 0, 0] = 0


def test_source_stays_open_until_last_user_stops():
    source = CountingCamera(fps=50)
    camera = SharedCamera(source)

    camera.start()
    camera.start()
    assert source.open_count == 1

    camera.stop()
    assert source.is_started()
    assert camera.capture() is not None

    camera.stop()
    assert not source.is_started()
    assert source.close_count == 1
    with pytest.raises(CameraReadError):
        camera.capture()

    camera.stop()  # Extra stops are harmless
    assert source.close_count == 1


def test_shared_camera_can_restart():
    source = CountingCamera(fps=50)
    camera = SharedCamera(source)

    with camera:
        assert camera.capture() is not None
    with camera:
        frame = camera.capture()

    assert frame is not None
    assert _frame_id(frame) == 2
    assert source.open_count == 2


def test_peripheral_release_stops_source_with_users_left():
    """App shutdown releases the source even if a user never stopped the shared camera."""
    Peripherals.clear()
    source = CountingCamera(fps=50)
    camera = SharedCamera(source)
    camera.start()
    camera.start()

    assert Peripherals.stop_all(timeout=2.0) == []

    assert not source.is_started()
    assert not camera.is_started()
    with pytest.raises(CameraReadError):
        camera.capture()


def test_waiting_consumer_wakes_up_on_stop():
    source = CountingCamera(fps=1)  # A source read blocks for up to a second
    camera = SharedCamera(source)
    errors: list[BaseException] = []

    def consumer():
        try:
            while True:
                camera.capture()
        except CameraReadError as e:
            errors.append(e)

    camera.start()
    threads = [threading.Thread(target=consumer, daemon=True) for _ in range(2)]
    for thread in threads:
        thread.start()
    time.sleep(0.1)  # One consumer reads from the source, the other waits for it

    start = time.monotonic()
    camera.stop()
    for thread in threads:
        thread.join(2.0)
        assert not thread.is_alive()

    assert time.monotonic() - start < 0.5
    assert len(errors) == 2


def test_source_read_error_does_not_hang_waiting_consumers():
    source = CountingCamera(fps=20, fail_read=True)
    camera = SharedCamera(source)
    errors: list[BaseException] = []
    barrier = threading.Barrier(2)

    def consumer():
        barrier.wait()
        try:
            camera.capture()
        except RuntimeError as e:
            errors.append(e)

    with camera:
        _run_in_threads(consumer, consumer)

    assert len(errors) == 2
