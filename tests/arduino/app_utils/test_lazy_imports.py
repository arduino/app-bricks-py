# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""Guards the app start time: importing arduino.app_utils must not load heavy dependencies the app may never use."""

import json
import subprocess
import sys
import textwrap
import unittest


def _loaded_after(code: str, modules: list[str]) -> dict[str, bool]:
    """Run code in a fresh interpreter and report which of the given modules it left in sys.modules."""
    probe = f"{code}\nimport json, sys\nprint(json.dumps({{m: m in sys.modules for m in {modules!r}}}))"
    result = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, timeout=120, check=True)
    return json.loads(result.stdout.strip().splitlines()[-1])


class TestAppUtilsLazyImports(unittest.TestCase):
    def test_import_does_not_load_heavy_dependencies(self):
        """The package imports every submodule: the submodules themselves must not load what their classes need."""
        heavy = ["requests", "urllib3", "numpy", "watchdog.observers"]
        for code in ("import arduino.app_utils", "from arduino.app_utils import *"):
            with self.subTest(code=code):
                self.assertEqual(_loaded_after(code, heavy), dict.fromkeys(heavy, False))

    def test_star_imported_classes_work_without_a_prior_import_of_their_dependencies(self):
        """Every method that needs numpy imports it itself."""
        code = textwrap.dedent("""
            import sys
            from arduino.app_utils import *

            assert "numpy" not in sys.modules and "requests" not in sys.modules
            block = SineGenerator(16000).generate_block(freq=440.0, amp_target=1.0, block_dur=0.01, master_volume=0.5)
            assert block.shape == (160,) and block.dtype == "float32"

            buffer = SlidingWindowBuffer(window_size=160, slide_amount=160)
            assert len(buffer.pull(timeout=0)) == 0
            assert buffer.push(block)
            assert (buffer.pull(timeout=0) == block).all()

            frame = Frame.from_rows([list(range(13)) for _ in range(8)])
            designer = FrameDesigner()
            designer.flip_horizontally(frame)
            assert frame.get_value(0, 0) == 12
            designer.rotate180(frame)
            designer.flip_vertically(frame)
            assert frame.get_value(0, 0) == 0
            assert len(frame.to_board_bytes()) == 8 * 13
        """)
        self.assertEqual(_loaded_after(code, ["numpy", "requests"]), {"numpy": True, "requests": False})

    def test_star_imported_folder_watcher_starts_its_observer(self):
        code = textwrap.dedent("""
            import tempfile
            from arduino.app_utils import *

            with tempfile.TemporaryDirectory() as path:
                watcher = FolderWatcher(path)
                watcher.start()
                watcher.stop()
        """)
        self.assertEqual(_loaded_after(code, ["watchdog.observers"]), {"watchdog.observers": True})

    def test_star_import_still_exports_every_public_name(self):
        code = "from arduino.app_utils import *\nimport arduino.app_utils as m\nassert all(n in globals() for n in m.__all__), m.__all__"
        subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120, check=True)


if __name__ == "__main__":
    unittest.main()
