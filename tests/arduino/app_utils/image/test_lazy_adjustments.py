# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""App start time: the image adjustments, which need OpenCV, load on first access; the PIL helpers do not need them."""

import inspect
import json
import subprocess
import sys

import pytest


def _loaded_after(code: str, modules: list[str]) -> dict[str, bool]:
    probe = f"{code}\nimport json, sys\nprint(json.dumps({{m: m in sys.modules for m in {modules!r}}}))"
    result = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, timeout=120, check=True)
    return json.loads(result.stdout.strip().splitlines()[-1])


@pytest.mark.parametrize(
    "code",
    [
        "import arduino.app_utils.image",
        "from arduino.app_utils.image import draw_bounding_boxes, draw_anomaly_markers, get_image_bytes, get_image_type, Shape, PipeableFunction",
    ],
)
def test_the_pil_helpers_do_not_load_opencv(code: str):
    assert _loaded_after(code, ["cv2", "arduino.app_utils.image.adjustments"]) == {"cv2": False, "arduino.app_utils.image.adjustments": False}


def test_an_adjustment_loads_on_first_access():
    loaded = _loaded_after(
        "import arduino.app_utils.image as image\nfrom arduino.app_utils.image import adjustments\nassert image.letterbox is adjustments.letterbox",
        ["cv2"],
    )
    assert loaded == {"cv2": True}


def test_every_public_adjustment_is_reachable_from_the_package():
    """Guards __all__ against a function added to adjustments.py and forgotten in the package."""
    import arduino.app_utils.image as image
    from arduino.app_utils.image import adjustments

    public = {
        name
        for name, value in vars(adjustments).items()
        if inspect.isfunction(value) and value.__module__ == adjustments.__name__ and not name.startswith("_")
    }
    assert public <= set(image.__all__)
    for name in public:
        assert getattr(image, name) is getattr(adjustments, name)


def test_star_import_still_exports_every_public_name():
    code = "from arduino.app_utils.image import *\nimport arduino.app_utils.image as m\nassert all(n in globals() for n in m.__all__), m.__all__"
    subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120, check=True)


def test_an_unknown_name_raises_attribute_error_without_loading_opencv():
    loaded = _loaded_after(
        "import arduino.app_utils.image as image\n"
        "try:\n    image.does_not_exist\nexcept AttributeError:\n    pass\nelse:\n    raise SystemExit('no AttributeError')",
        ["cv2"],
    )
    assert loaded == {"cv2": False}
