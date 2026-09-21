# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

import colorsys
import http.client
import threading
import urllib.error
import urllib.request

import numpy as np
import pytest

from arduino.app_bricks.video_objectdetection.video_stream import LabelColors, VideoStreamServer, draw_detections

JPEG_A = b"\xff\xd8A\xff\xd9"
JPEG_B = b"\xff\xd8BB\xff\xd9"


def read_part(response):
    """The next JPEG of a multipart/x-mixed-replace response."""
    length = None
    while True:
        line = response.readline()
        if line.lower().startswith(b"content-length:"):
            length = int(line.split(b":", 1)[1])
        if line == b"\r\n" and length is not None:
            return response.read(length)


@pytest.fixture
def stream():
    server = VideoStreamServer("127.0.0.1", 0)
    server.start()
    yield server
    server.stop()


def open_stream(server, path="/"):
    connection = http.client.HTTPConnection("127.0.0.1", server.port, timeout=5)
    connection.request("GET", path)
    response = connection.getresponse()
    return connection, response


def wait_for(cond, timeout=3.0):
    event = threading.Event()
    deadline = threading.Timer(timeout, event.set)
    deadline.start()
    try:
        while not cond() and not event.is_set():
            threading.Event().wait(0.02)
        return cond()
    finally:
        deadline.cancel()


def test_embed_page_shows_the_stream_through_an_img_tag(stream):
    with urllib.request.urlopen(f"http://127.0.0.1:{stream.port}/embed", timeout=5) as response:
        assert response.headers["Content-Type"].startswith("text/html")
        page = response.read()
        assert b'<img src="/"' in page
        assert b"width:100%;height:100%;object-fit:contain" in page, "the whole frame is scaled into the iframe"


def test_unknown_path_is_not_found(stream):
    with pytest.raises(urllib.error.HTTPError) as info:
        urllib.request.urlopen(f"http://127.0.0.1:{stream.port}/nope", timeout=5)
    assert info.value.code == 404


def test_clients_receive_the_frames_published_after_they_connect(stream):
    assert not stream.has_clients
    connection, response = open_stream(stream)
    assert response.status == 200 and response.headers["Content-Type"].startswith("multipart/x-mixed-replace")
    assert wait_for(lambda: stream.has_clients)
    stream.publish(JPEG_A)
    assert read_part(response) == JPEG_A
    stream.publish(JPEG_B)
    assert read_part(response) == JPEG_B
    response.close()  # the response holds its own handle on the socket
    connection.close()
    assert wait_for(lambda: not stream.has_clients), "a client that leaves is no longer counted"


def test_a_late_client_gets_the_latest_frame(stream):
    stream.publish(JPEG_A)
    stream.publish(JPEG_B)
    connection, response = open_stream(stream)
    assert read_part(response) == JPEG_B
    connection.close()


def test_stop_releases_the_connected_clients(stream):
    connection, response = open_stream(stream)
    assert wait_for(lambda: stream.has_clients)
    stream.stop()
    assert response.read() == b"", "the stream ends"
    connection.close()
    assert not stream.has_clients


# ---------------------------------------------------------------- drawing


def test_a_label_keeps_its_random_color_for_the_whole_run():
    colors = LabelColors()
    assert colors["cat"] == colors["cat"]
    assert colors["cat"] != colors["dog"]
    assert LabelColors(seed=1)["cat"] == LabelColors(seed=1)["cat"], "reproducible with a seed"
    assert all(0 <= c <= 255 for c in colors["cat"])


def test_the_labels_of_a_run_get_well_separated_hues():
    colors = LabelColors()
    hues = [colorsys.rgb_to_hsv(r / 255, g / 255, b / 255)[0] for b, g, r in (colors[f"label{i}"] for i in range(8))]
    for i, a in enumerate(hues):
        for b in hues[i + 1 :]:
            assert min(abs(a - b), 1 - abs(a - b)) > 0.04, (a, b)


def test_detections_are_drawn_as_a_box_with_a_label_chip_above():
    colors = LabelColors(seed=3)
    frame = np.zeros((480, 640, 3), np.uint8)
    detections = {"cat": [{"confidence": 0.57, "bounding_box_xyxy": (200, 200, 300, 400)}]}
    image = draw_detections(frame, detections, colors)
    assert frame.max() == 0, "the frame is not modified"
    color = colors["cat"]
    assert tuple(image[300, 200]) == color, "left border"
    assert tuple(image[300, 299]) == color, "right border"
    assert tuple(image[190, 210]) == color, "the chip above the box has the same color"
    assert (image[160:199, 200:300] >= 230).all(axis=2).any(), "white text in the chip"
    assert tuple(image[150, 250]) == (0, 0, 0), "the chip is compact"
    assert tuple(image[300, 250]) == (0, 0, 0), "the box is not filled"
    assert tuple(image[450, 500]) == (0, 0, 0), "nothing drawn far from the box"


def test_label_chip_moves_inside_the_box_when_there_is_no_room_above():
    colors = LabelColors(seed=3)
    image = draw_detections(np.zeros((480, 640, 3), np.uint8), {"cat": [{"confidence": 0.9, "bounding_box_xyxy": (10, 5, 200, 300)}]}, colors)
    assert tuple(image[15, 20]) == colors["cat"], "the chip starts at the top of the box"
