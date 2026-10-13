# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

import http.client
import threading
import urllib.error
import urllib.request
import zlib

import numpy as np
import pytest

from arduino.app_internal.edge_impulse.video_stream import PALETTE, BoxStabilizer, LabelColors, VideoStreamServer, draw_detections
from arduino.app_utils.image.colors import color_difference
from arduino.app_internal.ei_inference import Box

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


def test_a_label_takes_the_palette_color_its_name_hashes_to_for_the_whole_run():
    colors = LabelColors()

    assert colors["cup"] == PALETTE[zlib.crc32(b"cup") % len(PALETTE)]
    assert colors["cup"] == colors["cup"]
    assert LabelColors()["cup"] == colors["cup"], "the same color on every run"


def test_a_label_whose_color_is_taken_gets_the_next_free_one():
    start = zlib.crc32(b"bottle") % len(PALETTE)
    assert zlib.crc32(b"vase") % len(PALETTE) == start, "the two names hash to the same color"
    colors = LabelColors()

    assert colors["bottle"] == PALETTE[start]
    assert colors["vase"] == PALETTE[(start + 1) % len(PALETTE)]


def test_once_every_color_is_taken_the_palette_starts_again_from_its_first_color():
    colors = LabelColors()

    assert {colors[f"label{i}"] for i in range(len(PALETTE))} == set(PALETTE)
    assert colors["one more"] == PALETTE[0]
    assert colors["and another"] == PALETTE[1]


def test_the_colors_too_close_to_the_ones_to_avoid_are_left_out_of_the_palette():
    yellow, magenta = (0, 255, 255), (255, 0, 255)
    colors = LabelColors(avoid=(yellow, magenta))

    used = {colors[f"label{i}"] for i in range(len(PALETTE))}

    assert used < set(PALETTE)
    for color in used:
        for avoided in (yellow, magenta):
            assert color_difference(color, avoided) >= LabelColors.MIN_DIFFERENCE, color


def test_the_boxes_take_the_color_of_their_color_key():
    colors = LabelColors()
    detections = {
        "cup #1": [{"confidence": 0.9, "bounding_box_xyxy": (100, 200, 200, 400)}],
        "cup #2": [{"confidence": 0.9, "bounding_box_xyxy": (400, 200, 500, 400)}],
    }

    image = draw_detections(np.zeros((480, 640, 3), np.uint8), detections, colors, color_key=lambda label: label.split(" #")[0])

    assert tuple(image[300, 100]) == tuple(image[300, 400]) == colors["cup"]


def test_detections_are_drawn_as_a_box_with_a_label_chip_at_the_top_right():
    colors = LabelColors()
    frame = np.zeros((480, 640, 3), np.uint8)
    detections = {"cat": [{"confidence": 0.57, "bounding_box_xyxy": (200, 200, 300, 400)}]}
    image = draw_detections(frame, detections, colors)
    assert frame.max() == 0, "the frame is not modified"
    color = colors["cat"]
    assert tuple(image[300, 200]) == color, "left border"
    assert tuple(image[300, 299]) == color, "right border"
    assert tuple(image[190, 295]) == color, "the chip above the box, at its right corner, has the same color"
    assert (image[160:199, 200:300] >= 230).all(axis=2).any(), "white text in the chip"
    assert tuple(image[190, 210]) == (0, 0, 0), "the chip is as wide as its text, not as the box"
    assert tuple(image[150, 250]) == (0, 0, 0), "the chip is compact"
    assert tuple(image[300, 250]) == (0, 0, 0), "the box is not filled"
    assert tuple(image[450, 500]) == (0, 0, 0), "nothing drawn far from the box"


def test_label_chip_moves_inside_the_box_when_there_is_no_room_above():
    colors = LabelColors()
    image = draw_detections(np.zeros((480, 640, 3), np.uint8), {"cat": [{"confidence": 0.9, "bounding_box_xyxy": (10, 5, 200, 300)}]}, colors)
    assert tuple(image[15, 190]) == colors["cat"], "the chip starts at the top-right corner of the box"
    assert tuple(image[15, 20]) == (0, 0, 0), "and does not span the box"


# ---------------------------------------------------------------- box stabilizer


def box(label, score, x=100, y=100, w=50, h=50):
    return Box(label, score, x, y, w, h)


def test_a_box_missing_from_one_result_is_held_then_dropped():
    steady = BoxStabilizer()
    steady.update([box("cat", 0.9)], round_trip=0.01)
    steady.update([], round_trip=0.01)
    assert "cat" in steady.visible(), "held for a moment after the last match"
    steady._tracks[0].last_seen -= 1.0  # the hold elapsed
    assert steady.visible() == {}


def test_edge_noise_on_a_still_object_does_not_move_its_box():
    steady = BoxStabilizer()
    steady.update([box("cat", 0.9, x=100, w=200, h=200)], round_trip=0.01)
    for dx, dw in ((4, -5), (-5, 5), (2, 4), (-3, -2)):  # within 3% of a 200 px box
        steady.update([box("cat", 0.9, x=100 + dx, w=200 + dw, h=200)], round_trip=0.01)
        (cat,) = steady.visible()["cat"]
        assert cat["bounding_box_xyxy"] == (100, 100, 300, 300), "still, with the size it had"


def test_the_jitter_scales_with_the_box_so_a_small_box_still_moves_by_the_same_fraction():
    steady = BoxStabilizer()
    steady.update([box("cat", 0.9, x=100, w=20, h=20)], round_trip=0.01)
    steady.update([box("cat", 0.9, x=104, w=20, h=20)], round_trip=0.01)
    (cat,) = steady.visible()["cat"]
    assert cat["bounding_box_xyxy"][0] > 100, "4 px is a fifth of a 20 px box, movement, not noise"


def test_a_small_change_is_followed_slowly_and_a_large_one_at_once():
    steady = BoxStabilizer()
    steady.update([box("cat", 0.9, x=100, w=100, h=100)], round_trip=0.01)
    steady.update([box("cat", 0.9, x=110, w=100, h=100)], round_trip=0.01)
    small = steady.visible()["cat"][0]["bounding_box_xyxy"][0]
    steady.update([box("cat", 0.9, x=150, w=100, h=100)], round_trip=0.01)  # half the box, still overlapping
    (cat,) = steady.visible()["cat"]
    large = cat["bounding_box_xyxy"][0]
    assert 100 < small < 105, small
    assert 130 < large < 150, large


def test_the_displayed_score_is_smoothed():
    steady = BoxStabilizer()
    steady.update([box("cat", 0.9)], round_trip=0.01)
    steady.update([box("cat", 0.4)], round_trip=0.01)
    (cat,) = steady.visible()["cat"]
    assert cat["confidence"] == 0.8


def test_tracks_are_per_label_and_a_moved_object_is_followed_not_duplicated():
    steady = BoxStabilizer()
    steady.update([box("cat", 0.9), box("dog", 0.9)], round_trip=0.01)
    steady.update([box("cat", 0.9, x=110)], round_trip=0.01)
    visible = steady.visible()
    assert len(visible["cat"]) == 1 and len(visible["dog"]) == 1


def test_the_hold_follows_the_inference_time():
    steady = BoxStabilizer()
    steady.update([box("cat", 0.9)], round_trip=0.4)
    steady._tracks[0].last_seen -= 0.5
    assert "cat" in steady.visible(), "a slow model gets a longer hold, two inference times"
    steady._tracks[0].last_seen -= 0.4
    assert steady.visible() == {}
