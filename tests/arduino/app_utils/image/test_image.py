# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

import io

from PIL import Image

from arduino.app_utils.image.image import FONT_FILE, Shape, draw_anomaly_markers, draw_bounding_boxes, get_image_bytes, get_image_type, label_font


def _image() -> Image.Image:
    return Image.new("RGB", (64, 48), "white")


def test_bounding_boxes_accept_supported_unsupported_and_missing_shapes():
    detection = {"detection": [{"class_name": "cat", "bounding_box_xyxy": [4, 4, 20, 20], "confidence": 90}]}
    for shape in (Shape.CIRCLE, Shape.RECTANGLE, "hexagon", None):
        out = draw_bounding_boxes(_image(), detection, shape=shape)
        assert isinstance(out, Image.Image)


def test_bounding_boxes_accept_the_simple_label_dictionary():
    detection = {"cat": [{"bounding_box_xyxy": [4, 4, 20, 20], "confidence": 90}]}
    assert isinstance(draw_bounding_boxes(_image(), detection), Image.Image)


def test_no_detection_returns_the_image_or_none():
    image = _image()
    assert draw_bounding_boxes(image, None) is image
    assert draw_anomaly_markers(image, None) is None


def test_image_type_of_bytes_and_images():
    buf = io.BytesIO()
    _image().save(buf, "PNG")
    assert get_image_type(buf.getvalue()) == "png"
    assert get_image_type(_image()) is None  # an in-memory image has no format


def test_image_bytes_of_a_path(tmp_path):
    path = tmp_path / "image.png"
    path.write_bytes(b"png")
    assert get_image_bytes(path) == b"png"
    assert get_image_bytes(str(path)) == b"png"


def test_the_label_font_ships_with_the_library_under_its_own_license():
    assert FONT_FILE.exists() and FONT_FILE.with_suffix(".ttf.license").exists()
    font = label_font(12)
    assert font.getname()[0] == "Open Sans", "the static SemiBold instance names only its family"
    assert label_font(12) is font, "one face per size"
