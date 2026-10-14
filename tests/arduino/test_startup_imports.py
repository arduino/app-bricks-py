# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""Guards the app start time: what the imports every app starts with load.

Importing is most of an app's start time on the board, and one module-level import in a util is enough to slow down
every app: numpy costs ~0.3 s there, requests ~0.25 s. Each entry point below lists the third-party packages that
library code imports while it loads, the same in every environment, and a budget for the number of modules it
loads in all. A new package, or a module count over the budget, fails the test with the line that caused it.

Import a new dependency inside the code that uses it. If the entry point really needs it, measure the start time on
the board and update the entry point here in the same change. The check counts imports instead of timing them:
timings on CI machines vary too much to catch a few hundred milliseconds on the board.
"""

import json
import subprocess
import sys
from dataclasses import dataclass

import pytest


@dataclass(frozen=True)
class StartupImports:
    packages: frozenset[str]
    """Third-party packages that library code imports itself while the entry point loads."""
    max_modules: int
    """Modules loaded in all, stdlib and dependencies included, with some headroom over today's count.

    Unlike the packages, the count depends on the environment: what the dependencies import optionally is not the same
    in a dev venv, on CI or in the image (the web stack loads ~700 modules in a dev venv, ~600 in the image).
    """


STARTUP_IMPORTS = {
    # Every app
    "from arduino.app_utils import *": StartupImports(frozenset({"arduino.router_bridge", "watchdog"}), max_modules=110),
    # Apps with a camera: each backend loads its own dependencies (cv2, websockets, ...) when Camera() picks it
    "import arduino.app_peripherals.camera": StartupImports(frozenset({"arduino.router_bridge", "numpy", "watchdog"}), max_modules=215),
    # Apps with a web interface
    "import arduino.app_bricks.web_ui": StartupImports(
        frozenset({"arduino.router_bridge", "fastapi", "fastapi_socketio", "uvicorn", "watchdog"}), max_modules=770
    ),
    # Every AI brick: the Edge Impulse facade loads its HTTP client, PIL and OpenCV when an inference needs them
    "import arduino.app_internal.core": StartupImports(frozenset({"arduino.router_bridge", "watchdog"}), max_modules=110),
    # Apps with a microphone: each backend loads its own dependencies (websockets, asyncio, ...) when Microphone() picks it
    "from arduino.app_peripherals.microphone import Microphone": StartupImports(
        frozenset({"arduino.router_bridge", "numpy", "watchdog"}), max_modules=220
    ),
    # An image brick: PIL for its images, without OpenCV, which only the image adjustments need
    "import arduino.app_bricks.object_detection": StartupImports(frozenset({"arduino.router_bridge", "PIL", "watchdog"}), max_modules=165),
    # Cloud API bricks: the HTTP client loads with the first request
    "import arduino.app_bricks.air_quality_monitoring": StartupImports(frozenset({"arduino.router_bridge", "watchdog"}), max_modules=110),
    "import arduino.app_bricks.weather_forecast": StartupImports(frozenset({"arduino.router_bridge", "watchdog"}), max_modules=110),
    # Audio bricks: the microphone needs numpy; the ASR needs websockets (with asyncio and ssl) as soon as it starts
    "import arduino.app_bricks.audio_classification": StartupImports(frozenset({"arduino.router_bridge", "numpy", "watchdog"}), max_modules=225),
    "import arduino.app_bricks.asr": StartupImports(frozenset({"arduino.router_bridge", "numpy", "watchdog", "websockets"}), max_modules=345),
}

# Runs the entry point in a fresh interpreter and prints, as JSON, the third-party packages that library code
# (arduino.app_*) imported, each with the line that imported it first, and the number of modules it loaded
_PROBE = r"""
import json, sys

def package(name):
    parts = name.split(".")
    if parts[0] == "arduino":  # A namespace shared with other distributions, such as arduino.router_bridge
        return None if len(parts) < 2 or parts[1].startswith("app_") else "arduino." + parts[1]
    return None if parts[0] in sys.stdlib_module_names else parts[0]

imported_by = {}

class RecordLibraryImports:
    def find_spec(self, name, path=None, target=None):
        pkg = package(name)
        if pkg is not None and pkg not in imported_by:
            frame = sys._getframe(1)
            while frame is not None and frame.f_code.co_filename.startswith("<frozen"):  # The import machinery
                frame = frame.f_back
            importer = frame.f_globals.get("__name__", "") if frame is not None else ""
            if importer.startswith("arduino.app_"):
                imported_by[pkg] = f"{importer}:{frame.f_lineno}"
        return None  # Only watches: the regular finders import the module

sys.meta_path.insert(0, RecordLibraryImports())
before = set(sys.modules)
exec(sys.argv[1])
modules = len(set(sys.modules) - before)
print(json.dumps({"packages": {p: by for p, by in imported_by.items() if p in sys.modules}, "modules": modules}))
"""


def _probe(entry_point: str) -> tuple[dict[str, str], int]:
    result = subprocess.run([sys.executable, "-c", _PROBE, entry_point], capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr
    out = json.loads(result.stdout.strip().splitlines()[-1])
    return out["packages"], out["modules"]


@pytest.mark.parametrize("entry_point", STARTUP_IMPORTS)
def test_entry_point_imports_only_its_listed_packages(entry_point: str):
    expected = STARTUP_IMPORTS[entry_point].packages
    packages, _ = _probe(entry_point)

    new = {p: by for p, by in packages.items() if p not in expected}
    assert not new, (
        f"`{entry_point}` now loads {', '.join(f'{p} (imported by {by})' for p, by in sorted(new.items()))}: "
        "every app using it pays for that at start. Import it inside the code that uses it, or measure the start "
        "time on the board and add it to STARTUP_IMPORTS."
    )
    gone = sorted(expected - packages.keys())
    assert not gone, f"`{entry_point}` no longer imports {', '.join(gone)}: remove it from STARTUP_IMPORTS, so it cannot come back unnoticed."


@pytest.mark.parametrize("entry_point", STARTUP_IMPORTS)
def test_entry_point_stays_within_its_module_budget(entry_point: str):
    budget = STARTUP_IMPORTS[entry_point].max_modules
    _, modules = _probe(entry_point)

    assert modules <= budget, (
        f"`{entry_point}` loads {modules} modules, over its budget of {budget}: "
        f'run `python -X importtime -c "{entry_point}"` to see what grew, and raise the budget only after measuring '
        "the start time on the board."
    )
