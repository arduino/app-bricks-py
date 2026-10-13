# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""python-apps-base generates the ALSA wrappers (.asoundrc) at build time, which is only correct as long as the file is static."""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

IMAGE_DIR = Path(__file__).parents[2] / "containers" / "bricks" / "python-apps-base"
SCRIPT = IMAGE_DIR / "scripts" / "provision-alsa-devices.sh"
RUN_SH = IMAGE_DIR / "scripts" / "run.sh"

BASH = shutil.which("bash")


def _generate(home: Path, **env: str) -> str:
    assert BASH is not None
    subprocess.run([BASH, SCRIPT.as_posix()], env={**os.environ, **env, "HOME": home.as_posix()}, check=True, timeout=120)
    return (home / ".asoundrc").read_text()


@pytest.mark.skipif(BASH is None or sys.platform == "win32", reason="needs a POSIX bash")
def test_asoundrc_does_not_depend_on_the_runtime_environment(tmp_path: Path):
    """Same output whatever the user, the devices or the environment of the container: safe to bake into the image."""
    build = tmp_path / "build"
    runtime = tmp_path / "runtime"
    build.mkdir()
    runtime.mkdir()

    baked = _generate(build)
    at_start = _generate(runtime, USER="someone-else", ALSA_CARD="1", BOARD_NAME="other")

    assert baked == at_start
    assert "pcm.plug_card_9_dev_4_mic" in baked
    assert "pcm.plug_card_9_dev_4_spk" in baked


def test_image_bakes_asoundrc_and_run_sh_regenerates_it_only_when_missing():
    dockerfile = (IMAGE_DIR / "Dockerfile").read_text()
    assert "HOME=/home/app bash /provision-alsa-devices.sh" in dockerfile
    assert dockerfile.index("provision-alsa-devices.sh") < dockerfile.index("chown arduino:arduino -R /home/app")

    run_sh = RUN_SH.read_text()
    assert '[ -f "$HOME/.asoundrc" ] || bash /provision-alsa-devices.sh' in run_sh
    unconditional = [line for line in run_sh.splitlines() if line.strip() == "bash /provision-alsa-devices.sh"]
    assert unconditional == []
