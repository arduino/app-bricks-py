# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

import contextlib
import functools
import gc
import threading
import time
from collections.abc import Iterator

import numpy as np
import pytest

from arduino.app_peripherals.camera import BaseCamera, Camera, CameraReadError, CameraTransformError, CSICamera, V4LCamera
from arduino.app_peripherals.camera import camera as camera_factory
from arduino.app_peripherals.camera._sharing import CameraFollower, follow
from arduino.app_utils.peripheral_registry import Peripherals

from conftest import two_csi_cameras_only, two_v4l_cameras  # noqa: F401


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
        self._set_status("connected")

    def _close_camera(self):
        self.close_count += 1
        self._set_status("disconnected")

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


def _shared(leader: BaseCamera, followers: int) -> list[BaseCamera]:
    """The leader followed by the given number of follower handles on its device."""
    return [leader, *(follow(leader, None) for _ in range(followers))]


@contextlib.contextmanager
def _started(handles: list[BaseCamera]) -> Iterator[None]:
    """Start every handle, stopping them all on exit."""
    with contextlib.ExitStack() as stack:
        for handle in handles:
            stack.enter_context(handle)
        yield


def _assert_increasing(ids: list[int]) -> None:
    """A handle receives each frame once and in the order it was read."""
    assert ids == sorted(set(ids))


# Direct access, the behaviour of a camera that is not shared


def test_unshared_camera_frames_stay_writable():
    camera = CountingCamera(fps=50)
    with camera:
        frame = camera.capture()

    assert frame is not None
    frame[0, 0, 0] = 0  # Not shared, so nothing to protect


def test_camera_returns_to_direct_access_once_followers_are_gone():
    leader = CountingCamera(fps=50)
    follower = follow(leader, None)
    assert leader._get_hub() is not None

    del follower
    gc.collect()

    assert leader._get_hub() is None
    with leader:
        frame = leader.capture()
    assert frame is not None
    assert frame.flags.writeable


# Frame fan-out


@pytest.mark.parametrize("followers", [1, 2, 3])
def test_handles_capturing_together_receive_the_same_frames(followers: int):
    """Handles capturing together get the same frame objects, read once from the device."""
    leader = CountingCamera()
    handles = _shared(leader, followers)
    barrier = threading.Barrier(len(handles))
    received: list[list[np.ndarray]] = [[] for _ in handles]

    def consumer(index: int):
        for _ in range(8):
            barrier.wait()
            frame = handles[index].capture()
            assert frame is not None
            received[index].append(frame)

    with _started(handles):
        _run_in_threads(*(functools.partial(consumer, index) for index in range(len(handles))))

    assert [_frame_id(f) for f in received[0]] == list(range(1, 9))
    for frames in received[1:]:
        assert len(frames) == 8
        assert all(a is b for a, b in zip(received[0], frames))
    assert leader.read_count == 8


def test_handles_are_told_apart_even_on_the_same_thread():
    """A handle is a consumer on its own, whatever thread reads from it."""
    leader = CountingCamera(fps=50)
    follower = follow(leader, None)

    with leader, follower:
        first = leader.capture()
        shared = follower.capture()  # Same thread, other handle: gets the frame just read
        second = leader.capture()

    assert first is not None and shared is not None and second is not None
    assert shared is first
    assert _frame_id(second) == 2
    assert leader.read_count == 2


@pytest.mark.parametrize("repeated", [0, 1], ids=["leader", "follower"])
def test_same_handle_captured_repeatedly_on_one_thread(repeated: int):
    """Each capture of the same handle waits for a new frame, read at the camera FPS."""
    leader = CountingCamera(fps=20)
    handles = _shared(leader, 1)

    with _started(handles):
        start = time.monotonic()
        frames = [handles[repeated].capture() for _ in range(5)]
        elapsed = time.monotonic() - start

    assert [_frame_id(f) for f in frames if f is not None] == [1, 2, 3, 4, 5]
    assert leader.read_count == 5
    assert elapsed >= 4 / 20 * 0.9  # The first read is immediate, the next four wait a frame interval each


@pytest.mark.parametrize(
    ("schedule", "expected_ids"),
    [
        ([0, 1, 2, 3] * 3, [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3]),
        ([0, 0, 1, 1, 2, 2, 3, 3], [1, 2, 2, 3, 3, 4, 4, 5]),
        ([3, 3, 3, 0, 1, 2, 0], [1, 2, 3, 3, 3, 3, 4]),
    ],
    ids=["round-robin", "each-twice", "one-handle-first"],
)
def test_handles_captured_in_turn_on_one_thread(schedule: list[int], expected_ids: list[int]):
    """
    A handle asking again reads a new frame, which the others then reuse while it is fresh.

    Every capture follows the previous one well within the frame interval, so the frame just read
    is always fresh for a handle that has not received it yet.
    """
    leader = CountingCamera(fps=10)
    handles = _shared(leader, 3)

    with _started(handles):
        frames = [handles[index].capture() for index in schedule]

    assert [_frame_id(f) for f in frames if f is not None] == expected_ids
    assert leader.read_count == max(expected_ids)
    # Handles given the same frame id got the very same object
    by_id: dict[int, np.ndarray] = {}
    for frame in frames:
        assert frame is not None
        assert by_id.setdefault(_frame_id(frame), frame) is frame


@pytest.mark.parametrize("followers", [1, 2, 3])
def test_streams_on_different_handles_receive_every_frame(followers: int):
    leader = CountingCamera()
    handles = _shared(leader, followers)
    barrier = threading.Barrier(len(handles))
    received: list[list[int]] = [[] for _ in handles]

    def consumer(index: int):
        for frame in handles[index].stream():
            received[index].append(_frame_id(frame))
            if len(received[index]) == 5:
                break
            barrier.wait()

    with _started(handles):
        _run_in_threads(*(functools.partial(consumer, index) for index in range(len(handles))))

    assert received == [[1, 2, 3, 4, 5]] * len(handles)


@pytest.mark.parametrize("followers", [1, 2, 3])
def test_handle_never_receives_a_frame_twice(followers: int):
    leader = CountingCamera(fps=50)
    handles = _shared(leader, followers)
    work = [0, 0.1, 0.03, 0.2][: len(handles)]  # Seconds spent on each frame, the leader being the fastest
    received: list[list[int]] = [[] for _ in handles]

    def consumer(index: int):
        deadline = time.monotonic() + 0.6
        while time.monotonic() < deadline:
            frame = handles[index].capture()
            if frame is not None:
                received[index].append(_frame_id(frame))
            time.sleep(work[index])

    with _started(handles):
        _run_in_threads(*(functools.partial(consumer, index) for index in range(len(handles))))

    for ids in received:
        _assert_increasing(ids)
        assert all(frame_id <= leader.read_count for frame_id in ids)
    # A slow handle skips the frames it was too slow to see instead of queuing them
    for index in range(1, len(handles)):
        assert len(received[index]) < len(received[0])


def test_hammered_handle_does_not_starve_the_streaming_ones():
    """A handle captured in a tight loop leaves the frames it reads to the handles streaming alongside."""
    leader = CountingCamera(fps=20)
    handles = _shared(leader, 3)
    received: list[list[int]] = [[] for _ in handles]
    done = threading.Event()

    def hammer():
        while not done.is_set():
            frame = leader.capture()
            if frame is not None:
                received[0].append(_frame_id(frame))

    def streamer(index: int):
        for frame in handles[index].stream():
            received[index].append(_frame_id(frame))
            if done.is_set():
                break

    with _started(handles):
        threads = [threading.Thread(target=hammer, daemon=True)]
        threads += [threading.Thread(target=streamer, args=(index,), daemon=True) for index in range(1, len(handles))]
        for thread in threads:
            thread.start()
        time.sleep(0.6)
        done.set()
        for thread in threads:
            thread.join(2.0)
            assert not thread.is_alive(), "consumer thread hung"

    for ids in received:
        _assert_increasing(ids)
    for ids in received[1:]:
        assert len(ids) >= leader.read_count // 2


def test_slow_handle_gets_the_latest_frame_without_waiting():
    leader = CountingCamera(fps=5)
    follower = follow(leader, None)

    with leader, follower:
        ids = [_frame_id(f) for f in (leader.capture() for _ in range(3)) if f is not None]

        start = time.monotonic()
        frame = follower.capture()  # Fresh frame read by the leader
        elapsed = time.monotonic() - start

    assert frame is not None
    assert _frame_id(frame) == ids[-1]
    assert elapsed < 0.1  # Well below the 200 ms frame interval
    assert leader.read_count == 3


def test_stale_frame_is_not_handed_out():
    leader = CountingCamera(fps=20)
    follower = follow(leader, None)

    with leader, follower:
        leader.capture()
        time.sleep(0.1)  # Two frame intervals
        frame = follower.capture()

    assert frame is not None
    assert _frame_id(frame) == 2
    assert leader.read_count == 2


def test_shared_frames_are_read_only():
    leader = CountingCamera(fps=50)
    follower = follow(leader, None)

    with leader, follower:
        frame = follower.capture()

    assert frame is not None
    with pytest.raises(ValueError):
        frame[0, 0, 0] = 0


def test_adjustments_are_per_handle():
    leader = CountingCamera(fps=50)
    leader.adjustments = lambda frame: frame * 10
    follower = follow(leader, lambda frame: frame + 1)
    plain = follow(leader, None)

    with leader, follower, plain:
        from_leader = leader.capture()
        from_follower = follower.capture()
        from_plain = plain.capture()

    assert from_leader is not None and from_follower is not None and from_plain is not None
    assert (_frame_id(from_leader), _frame_id(from_follower), _frame_id(from_plain)) == (10, 2, 1)


def test_in_place_adjustment_fails_loudly():
    """An adjustment writing in place would corrupt the other handles' frame: it raises instead."""
    leader = CountingCamera(fps=50)

    def in_place(frame: np.ndarray) -> np.ndarray:
        frame += 1
        return frame

    follower = follow(leader, in_place)

    with leader, follower:
        with pytest.raises(CameraTransformError):
            follower.capture()
        frame = leader.capture()

    assert frame is not None
    assert _frame_id(frame) == 1


# Lifecycle


def test_device_stays_open_until_last_handle_stops():
    leader = CountingCamera(fps=50)
    follower = follow(leader, None)

    leader.start()
    follower.start()
    assert leader.open_count == 1

    leader.stop()
    assert not leader.is_started()
    assert follower.is_started()
    assert follower.capture() is not None
    with pytest.raises(CameraReadError):
        leader.capture()

    follower.stop()
    assert not follower.is_started()
    assert leader.close_count == 1
    with pytest.raises(CameraReadError):
        follower.capture()

    follower.stop()  # Extra stops are harmless
    leader.stop()
    assert leader.close_count == 1


def test_follower_started_first_opens_the_device():
    leader = CountingCamera(fps=50)
    follower = follow(leader, None)

    follower.start()
    assert leader.open_count == 1
    assert follower.resolution == (8, 6)  # The resolution negotiated by the device
    assert not leader.is_started()
    with pytest.raises(CameraReadError):
        leader.capture()

    leader.start()
    assert leader.open_count == 1
    assert leader.capture() is not None

    follower.stop()
    leader.stop()
    assert leader.close_count == 1


def test_follower_created_on_a_started_camera():
    leader = CountingCamera(fps=50)
    leader.start()
    assert leader.capture() is not None

    follower = follow(leader, None)
    assert not follower.is_started()
    follower.start()
    frame = follower.capture()

    assert frame is not None
    assert leader.open_count == 1
    assert leader.is_started()
    leader.stop()
    follower.stop()


def test_shared_device_can_restart():
    leader = CountingCamera(fps=50)
    follower = follow(leader, None)

    with leader, follower:
        assert follower.capture() is not None
    with follower:
        frame = follower.capture()

    assert frame is not None
    assert _frame_id(frame) == 2
    assert leader.open_count == 2


def test_peripheral_release_closes_a_shared_device():
    """App shutdown releases the device, even if a started handle was garbage collected."""
    Peripherals.clear()
    leader = CountingCamera(fps=50)
    follower = follow(leader, None)
    other = follow(leader, None)
    leader.start()
    follower.start()
    other.start()

    del other
    gc.collect()
    assert Peripherals.stop_all(timeout=2.0) == []

    assert leader.close_count == 1
    assert not follower.is_started()
    with pytest.raises(CameraReadError):
        follower.capture()


def test_waiting_handle_wakes_up_on_stop():
    leader = CountingCamera(fps=1)  # A device read blocks for up to a second
    follower = follow(leader, None)
    errors: list[BaseException] = []

    def consumer(handle: BaseCamera):
        try:
            while True:
                handle.capture()
        except CameraReadError as e:
            errors.append(e)

    leader.start()
    follower.start()
    threads = [threading.Thread(target=consumer, args=(handle,), daemon=True) for handle in (leader, follower)]
    for thread in threads:
        thread.start()
    time.sleep(0.1)  # One handle reads from the device, the other waits for it

    start = time.monotonic()
    leader.stop()
    follower.stop()
    for thread in threads:
        thread.join(2.0)
        assert not thread.is_alive()

    assert time.monotonic() - start < 0.5
    assert len(errors) == 2


def test_handles_stopped_one_by_one_while_the_others_stream():
    """App stops its bricks one at a time: the handles still started keep streaming meanwhile."""
    leader = CountingCamera(fps=20)
    handles = _shared(leader, 3)
    received: list[list[tuple[int, float]]] = [[] for _ in handles]
    ended = [0.0] * len(handles)

    def streamer(index: int):
        try:
            for frame in handles[index].stream():
                received[index].append((_frame_id(frame), time.monotonic()))
        except CameraReadError:
            pass  # Stopped between the stream's started check and its capture
        ended[index] = time.monotonic()

    for handle in handles:
        handle.start()
    threads = [threading.Thread(target=streamer, args=(index,), daemon=True) for index in range(len(handles))]
    for thread in threads:
        thread.start()

    stop_order = [2, 0, 3, 1]  # A follower first, then the leader, so that the device outlives its owner
    stopped_at: dict[int, float] = {}
    for index in stop_order:
        time.sleep(0.25)
        stopped_at[index] = time.monotonic()
        handles[index].stop()
        threads[index].join(1.0)
        assert not threads[index].is_alive(), f"handle {index} kept streaming after its stop"
        assert ended[index] - stopped_at[index] < 0.5
        assert leader.close_count == (1 if len(stopped_at) == len(handles) else 0)

    assert leader.open_count == 1
    for frames in received:
        _assert_increasing([frame_id for frame_id, _ in frames])
    # Each stop left the handles still started streaming
    for position, index in enumerate(stop_order):
        for later in stop_order[position + 1 :]:
            assert any(at > stopped_at[index] for _, at in received[later]), f"handle {later} starved after the stop of {index}"


def test_parallel_release_with_captures_in_flight():
    """Releasing every handle at once, as the app shutdown does, while their threads are in capture()."""
    Peripherals.clear()
    leader = CountingCamera(fps=1)  # One handle sleeps in the FPS throttle for up to a second
    handles = _shared(leader, 3)
    errors: list[BaseException] = []

    def consumer(index: int):
        try:
            while True:
                handles[index].capture()
        except CameraReadError as e:
            errors.append(e)

    for handle in handles:
        handle.start()
    threads = [threading.Thread(target=consumer, args=(index,), daemon=True) for index in range(len(handles))]
    for thread in threads:
        thread.start()
    time.sleep(0.2)  # One handle reads the second frame from the device, the others wait for it

    start = time.monotonic()
    assert Peripherals.stop_all(timeout=2.0) == []
    elapsed = time.monotonic() - start
    for thread in threads:
        thread.join(1.0)
        assert not thread.is_alive()

    assert elapsed < 0.5  # The throttle sleep is cut short rather than waited out
    assert leader.close_count == 1
    assert len(errors) == len(handles)

    # The device opens again for the next run
    with _started(handles):
        assert handles[3].capture() is not None
    assert leader.open_count == 2


def test_device_read_error_does_not_hang_waiting_handles():
    leader = CountingCamera(fps=20, fail_read=True)
    follower = follow(leader, None)
    errors: list[BaseException] = []
    barrier = threading.Barrier(2)

    def consumer(handle: BaseCamera):
        barrier.wait()
        try:
            handle.capture()
        except RuntimeError as e:
            errors.append(e)

    with leader, follower:
        _run_in_threads(lambda: consumer(leader), lambda: consumer(follower))

    assert len(errors) == 2


def test_status_changes_reach_every_handle():
    leader = CountingCamera(fps=50)
    follower = follow(leader, None)
    seen: dict[str, list[str]] = {"leader": [], "follower": []}
    done = threading.Event()

    leader.on_status_changed(lambda status, data: seen["leader"].append(status))

    def on_follower_status(status: str, data: dict):
        seen["follower"].append(status)
        if status == "streaming":
            done.set()

    follower.on_status_changed(on_follower_status)

    with follower:
        follower.capture()
        assert done.wait(2.0)

    assert follower.status == leader.status
    assert seen["follower"][:2] == ["connected", "streaming"]


# Factory


@pytest.fixture
def fake_v4l_stream(monkeypatch):
    """Make V4L cameras stream counted frames instead of opening a device."""
    counters: dict[str, int] = {}
    opened: list[str] = []

    def open_camera(self):
        opened.append(self.v4l_path)
        self._set_status("connected")

    def read_frame(self):
        counters[self.v4l_path] = counters.get(self.v4l_path, 0) + 1
        return np.full((2, 2, 3), counters[self.v4l_path], dtype=np.int32)

    monkeypatch.setattr(V4LCamera, "_open_camera", open_camera)
    monkeypatch.setattr(V4LCamera, "_close_camera", lambda self: None)
    monkeypatch.setattr(V4LCamera, "_read_frame", read_frame)
    return opened


@pytest.mark.parametrize("count", [2, 4])
def test_factory_shares_a_camera_selected_several_times(two_v4l_cameras, fake_v4l_stream, count: int):
    handles = [Camera(0, fps=50) for _ in range(count)]
    first = handles[0]

    assert isinstance(first, V4LCamera)  # A camera used once is unchanged
    for handle in handles[1:]:
        assert isinstance(handle, CameraFollower)
        assert handle.leader is first
        assert handle.v4l_path == first.v4l_path  # Camera-specific attributes come from the leader

    with _started(handles):
        frames = [handle.capture() for handle in handles]

    assert frames[0] is not None
    assert all(frame is frames[0] for frame in frames)
    assert fake_v4l_stream == [first.v4l_path]  # Opened once


def test_factory_shares_with_an_auto_selected_camera(two_v4l_cameras, fake_v4l_stream):
    auto = Camera()
    explicit = Camera(0)

    assert isinstance(explicit, CameraFollower)
    assert explicit.leader is auto


def test_factory_auto_selection_still_picks_distinct_cameras(two_v4l_cameras, fake_v4l_stream):
    cam1 = Camera()
    cam2 = Camera()

    assert isinstance(cam1, V4LCamera) and isinstance(cam2, V4LCamera)
    assert cam1.v4l_path != cam2.v4l_path


def test_factory_auto_selection_skips_a_shared_camera(two_v4l_cameras, fake_v4l_stream):
    first = Camera(0)
    shared = Camera(0)
    auto = Camera()

    del first
    gc.collect()  # The follower keeps the device claimed

    assert shared.v4l_path == "/dev/v4l/by-id/usb-CamA-video-index0"
    assert auto.v4l_path == "/dev/v4l/by-id/usb-CamB-video-index0"


@pytest.fixture
def factory_warnings(monkeypatch):
    """Collect the warnings logged by the Camera factory."""
    warnings: list[str] = []
    monkeypatch.setattr(camera_factory.logger, "warning", lambda message, *args, **kwargs: warnings.append(message))
    return warnings


def test_factory_follower_uses_the_first_configuration(two_v4l_cameras, fake_v4l_stream, factory_warnings):
    first = Camera(0, resolution=(640, 480), fps=10)
    second = Camera(0, resolution=(1280, 720), fps=30)

    assert (second.resolution, second.fps) == ((640, 480), 10)
    assert first.resolution == (640, 480)
    assert len(factory_warnings) == 1
    assert "already in use" in factory_warnings[0]


def test_factory_follower_with_the_same_configuration_does_not_warn(two_v4l_cameras, fake_v4l_stream, factory_warnings):
    Camera(0, fps=15)
    Camera(0, fps=15)

    assert factory_warnings == []


def test_factory_shares_csi_cameras(two_csi_cameras_only):
    first = Camera("csi:0")
    second = Camera("csi:CAMERA0")
    other = Camera("csi:1")

    assert isinstance(first, CSICamera)
    assert isinstance(second, CameraFollower)
    assert second.csi_path == first.csi_path
    assert isinstance(other, CSICamera)
