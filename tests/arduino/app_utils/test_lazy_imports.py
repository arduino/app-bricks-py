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
        loaded = _loaded_after("import arduino.app_utils", ["requests", "numpy", "watchdog"])
        self.assertEqual(loaded, {"requests": False, "numpy": False, "watchdog": False})

    def test_lazy_names_resolve_to_their_submodule_classes(self):
        import arduino.app_utils as app_utils
        from arduino.app_utils import audio, folderwatch, httprequest, ledmatrix, slidingwindowbuffer

        self.assertIs(app_utils.SineGenerator, audio.SineGenerator)
        self.assertIs(app_utils.FolderWatcher, folderwatch.FolderWatcher)
        self.assertIs(app_utils.FolderEventHandler, folderwatch.FolderEventHandler)
        self.assertIs(app_utils.HttpClient, httprequest.HttpClient)
        self.assertIs(app_utils.Frame, ledmatrix.Frame)
        self.assertIs(app_utils.FrameDesigner, ledmatrix.FrameDesigner)
        self.assertIs(app_utils.SlidingWindowBuffer, slidingwindowbuffer.SlidingWindowBuffer)

    def test_from_import_loads_on_demand(self):
        loaded = _loaded_after("from arduino.app_utils import HttpClient", ["arduino.app_utils.httprequest", "numpy"])
        self.assertEqual(loaded, {"arduino.app_utils.httprequest": True, "numpy": False})

    def test_star_import_does_not_load_heavy_dependencies(self):
        """A star import resolves every lazy name: the submodules themselves must not load what the classes need."""
        loaded = _loaded_after("from arduino.app_utils import *", ["requests", "numpy", "watchdog.observers"])
        self.assertEqual(loaded, {"requests": False, "numpy": False, "watchdog.observers": False})

    def test_star_imported_numpy_classes_work_without_a_prior_numpy_import(self):
        """Every method that needs numpy imports it itself."""
        code = textwrap.dedent("""
            import sys
            from arduino.app_utils import *

            assert "numpy" not in sys.modules
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
        self.assertEqual(_loaded_after(code, ["numpy"]), {"numpy": True})

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

    def test_resolved_name_is_cached_in_the_package(self):
        loaded = _loaded_after(
            "import arduino.app_utils as m\nassert 'HttpClient' not in vars(m)\nm.HttpClient\nassert 'HttpClient' in vars(m)",
            ["arduino.app_utils.httprequest"],
        )
        self.assertEqual(loaded, {"arduino.app_utils.httprequest": True})

    def test_dir_lists_the_lazy_names_without_loading_them(self):
        loaded = _loaded_after(
            "import arduino.app_utils as m\nassert {'HttpClient', 'SineGenerator', 'App'} <= set(dir(m))",
            ["requests", "numpy"],
        )
        self.assertEqual(loaded, {"requests": False, "numpy": False})

    def test_unknown_attribute_raises_attribute_error(self):
        import arduino.app_utils as app_utils

        with self.assertRaises(AttributeError):
            _ = app_utils.DoesNotExist  # pyright: ignore[reportAttributeAccessIssue]


if __name__ == "__main__":
    unittest.main()
