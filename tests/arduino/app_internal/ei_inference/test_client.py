# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

import numpy as np
import pytest

from arduino.app_internal.ei_inference import InferenceClient, ServerError, model_name_from_path

FRAME = np.zeros((100, 200, 3), np.uint8)  # 200x100 source, twice as wide as the 100x100 model


ROOTS = ("/var/lib/arduino-app-cli/models", "/models")


def test_model_name_is_the_path_relative_to_the_models_directory():
    root = ROOTS[0]
    assert model_name_from_path(f"{root}/custom-ei/abc/model.eim", ROOTS) == "custom-ei/abc/model"
    assert model_name_from_path(f"{root}/edge-impulse/yolo-x-nano/yolo-x-nano.eim", ROOTS) == "edge-impulse/yolo-x-nano/yolo-x-nano"
    assert model_name_from_path(f"{root}//edge-impulse/../edge-impulse/x.eim", ROOTS) == "edge-impulse/x"
    assert model_name_from_path("/models/ootb/ei/yolo-x-nano.eim", ROOTS) == "ootb/ei/yolo-x-nano", "a model bundled in the image"


@pytest.mark.parametrize(
    "path", ["/home/arduino/model.eim", "/var/lib/arduino-app-cli/models/x/model.tflite", "/var/lib/arduino-app-cli/models", "/models"]
)
def test_model_name_rejects_paths_outside_the_models_directories(path):
    with pytest.raises(ValueError, match="inference service"):
        model_name_from_path(path, ROOTS)


def test_open_unknown_model_raises_server_error(ei_service):
    with pytest.raises(ServerError) as info:
        InferenceClient("nope", ei_service.socket_path)
    assert info.value.code == "unknown_model"


def test_open_exposes_the_model_details(ei_service):
    with InferenceClient("det", ei_service.socket_path) as client:
        assert client.model == "det"
        assert client.labels == ["cat", "dog"]
        assert client.input_size == (100, 100)
        assert client.info["resize_mode"] == "squash"


def test_infer_sends_the_frame_as_it_is_and_returns_the_boxes(ei_service):
    ei_service.models["det"]["boxes"] = [{"label": "cat", "score": 0.9, "x": 50, "y": 25, "w": 100, "h": 50}]
    with InferenceClient("det", ei_service.socket_path) as client:
        assert client.resize_mode == "squash"
        result = client.infer(FRAME, timeout=5)
    assert result.ok and result.seq == 1 and result.source_size == (200, 100)
    assert ei_service.frames[0][2].shape == (100, 200, 3), "the frame reaches the service at the camera size"
    box = result.boxes[0]
    assert (box.label, box.score) == ("cat", 0.9)
    assert (box.x, box.y, box.w, box.h) == (50, 25, 100, 50), "boxes are taken as the service maps them"


def test_frames_carry_their_channel_order(ei_service):
    frames = []
    ei_service.models["det"]["boxes"] = lambda seq, image: frames.append(image.copy()) or []
    with InferenceClient("det", ei_service.socket_path) as client:
        client.infer(FRAME, color="bgr", timeout=5)
        client.infer(FRAME, color="rgb", timeout=5)
    assert len(frames) == 2, "both frames reached the service unchanged, the color travels in the header"


def test_frame_error_is_a_failed_result_and_the_connection_stays_usable(ei_service):
    replies = iter([{"code": "internal", "error": "boom"}, []])
    ei_service.models["det"]["boxes"] = lambda seq, image: next(replies)
    with InferenceClient("det", ei_service.socket_path) as client:
        failed = client.infer(FRAME, timeout=5)
        assert not failed.ok and failed.error_code == "internal" and failed.error == "boom"
        assert client.infer(FRAME, timeout=5).ok


def test_keep_frame_returns_a_copy_of_the_submitted_image(ei_service):
    with InferenceClient("det", ei_service.socket_path) as client:
        source = FRAME.copy()
        result = client.infer(source, keep_frame=True, timeout=5)
        source[:] = 255
    assert result.frame is not None and result.frame.max() == 0


def test_submit_skips_frames_while_one_is_in_flight(ei_service):
    ei_service.reply_delay = 0.3
    with InferenceClient("det", ei_service.socket_path) as client:
        assert client.submit(FRAME) == 1
        assert client.busy
        assert client.submit(FRAME) is None
        assert client.skipped == 1
        result = client.get_result(timeout=5)
        assert result is not None and result.seq == 1
        assert client.latest is result
        assert client.get_result(timeout=0) is None, "a result is returned once"


def test_wait_idle_tells_when_the_model_is_free(ei_service):
    ei_service.reply_delay = 0.2
    with InferenceClient("det", ei_service.socket_path) as client:
        assert client.wait_idle(0) is True, "free before anything is sent"
        assert client.submit(FRAME) == 1
        assert client.wait_idle(0.05) is False, "still busy after 50 ms"
        assert client.wait_idle(1.0) is True, "free once the reply arrived"
        assert client.submit(FRAME) == 2
    with pytest.raises(ConnectionError):
        client.wait_idle(0.1)


def test_infer_times_out_when_the_reply_does_not_arrive(ei_service):
    ei_service.reply_delay = 2.0
    with InferenceClient("det", ei_service.socket_path) as client:
        with pytest.raises(TimeoutError):
            client.infer(FRAME, timeout=0.2)


def test_close_releases_the_connection(ei_service):
    client = InferenceClient("det", ei_service.socket_path)
    client.close()
    assert ei_service.closed.wait(2), "the service sees the connection close"
    with pytest.raises(ConnectionError):
        client.submit(FRAME)


def test_service_going_away_closes_the_client(ei_service):
    client = InferenceClient("det", ei_service.socket_path)
    ei_service.stop()
    with pytest.raises(ConnectionError):
        client.get_result(timeout=5)
    assert client.closed
