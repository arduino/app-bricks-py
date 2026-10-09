# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""Guards the app start time: importing the camera package must not load the dependencies of backends the app does not use."""

import json
import subprocess
import sys
import unittest


def _loaded_after(code: str, modules: list[str]) -> dict[str, bool]:
    """Run code in a fresh interpreter and report which of the given modules it left in sys.modules."""
    probe = f"{code}\nimport json, sys\nprint(json.dumps({{m: m in sys.modules for m in {modules!r}}}))"
    result = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, timeout=120, check=True)
    return json.loads(result.stdout.strip().splitlines()[-1])


class TestCameraLazyBackends(unittest.TestCase):
    def test_import_does_not_load_backend_dependencies(self):
        loaded = _loaded_after(
            "from arduino.app_peripherals.camera import Camera, BaseCamera",
            ["cv2", "requests", "websockets", "cryptography"],
        )
        self.assertEqual(loaded, {"cv2": False, "requests": False, "websockets": False, "cryptography": False})

    def test_backends_resolve_to_their_submodule_classes(self):
        import arduino.app_peripherals.camera as camera
        from arduino.app_peripherals.camera import ip_camera, websocket_camera

        self.assertIs(camera.IPCamera, ip_camera.IPCamera)
        self.assertIs(camera.WebSocketCamera, websocket_camera.WebSocketCamera)

    @unittest.skipUnless(sys.platform.startswith("linux"), "V4L and CSI backends need fcntl")
    def test_linux_backends_resolve_to_their_submodule_classes(self):
        import arduino.app_peripherals.camera as camera
        from arduino.app_peripherals.camera import csi_camera, v4l_camera

        self.assertIs(camera.V4LCamera, v4l_camera.V4LCamera)
        self.assertIs(camera.CSICamera, csi_camera.CSICamera)


if __name__ == "__main__":
    unittest.main()
