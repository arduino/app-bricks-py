# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""python-apps-base strips the debug sections of the compiled extensions it installs: the stripped module must still
work, and a file that carries none must be left alone, a rewrite would copy it out of the python-base layers."""

import os
import shutil
import subprocess
import sys
from importlib.util import find_spec
from pathlib import Path

import pytest

IMAGE_DIR = Path(__file__).parents[2] / "containers" / "bricks" / "python-apps-base"
IMAGE_SITE_PACKAGES = "/usr/local/lib/python3.13/site-packages"

STRIP_COMMAND = (
    'find {site_packages} -type f \\( -name "*.so" -o -name "*.so.*" \\) '
    """-exec sh -c 'readelf -S "$1" | grep -q " \\.debug_"' sh {{}} \\; -print -exec strip --strip-debug {{}} +"""
)

needs_binutils = pytest.mark.skipif(
    sys.platform != "linux" or shutil.which("strip") is None or shutil.which("readelf") is None,
    reason="needs GNU binutils on Linux, as the image build",
)


def _sections(shared_object: Path) -> str:
    return subprocess.run(["readelf", "-S", "-W", str(shared_object)], capture_output=True, text=True, check=True, timeout=60).stdout


def _strip(site_packages: Path) -> list[str]:
    """Run the command of the image build on site_packages and return the files it reports as stripped."""
    command = STRIP_COMMAND.format(site_packages=site_packages)
    return subprocess.run(["sh", "-ec", command], capture_output=True, text=True, check=True, timeout=120).stdout.split()


def test_image_strips_with_the_tested_command_before_compiling_and_drops_the_tool():
    dockerfile = (IMAGE_DIR / "Dockerfile").read_text()
    strip = STRIP_COMMAND.format(site_packages=IMAGE_SITE_PACKAGES)
    assert strip in dockerfile
    assert dockerfile.index("apt-get install -y --no-install-recommends --no-install-suggests patch binutils") < dockerfile.index(strip)
    assert dockerfile.index(strip) < dockerfile.index("apt-get purge -y --auto-remove patch binutils")
    assert dockerfile.index(strip) < dockerfile.index("python -m compileall /usr/local/lib")


@needs_binutils
def test_a_stripped_extension_still_works_and_is_not_rewritten_again(tmp_path: Path):
    spec = find_spec("uvloop")
    assert spec is not None and spec.submodule_search_locations, "uvloop is not installed, run `uv sync`"
    site_packages = tmp_path / "site-packages"
    shutil.copytree(spec.submodule_search_locations[0], site_packages / "uvloop", ignore=shutil.ignore_patterns("__pycache__"))
    extension = next((site_packages / "uvloop").glob("loop.*.so"))
    if " .debug_" not in _sections(extension):
        pytest.skip("the uvloop wheel of this platform ships no debug sections")
    size = extension.stat().st_size

    assert _strip(site_packages) == [str(extension)]

    sections = _sections(extension)
    assert " .debug_" not in sections
    assert " .symtab" in sections and " .dynsym" in sections
    assert extension.stat().st_size < size

    stripped = extension.stat()
    assert _strip(site_packages) == []
    assert (extension.stat().st_ino, extension.stat().st_mtime_ns) == (stripped.st_ino, stripped.st_mtime_ns)

    code = (
        "import asyncio, sys, uvloop\n"
        "assert uvloop.__file__.startswith(sys.argv[1]), uvloop.__file__\n"
        "async def main():\n"
        "    await asyncio.sleep(0.01)\n"
        "    return type(asyncio.get_running_loop()).__module__\n"
        "assert uvloop.run(main()) == 'uvloop'\n"
    )
    pythonpath = [str(site_packages), *filter(None, os.environ.get("PYTHONPATH", "").split(os.pathsep))]
    result = subprocess.run(
        [sys.executable, "-X", "faulthandler", "-c", code, str(site_packages)],
        env={**os.environ, "PYTHONPATH": os.pathsep.join(pythonpath)},
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
