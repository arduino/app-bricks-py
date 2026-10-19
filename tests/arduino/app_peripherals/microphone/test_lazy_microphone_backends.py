# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""Guards the app start time: importing the microphone package must not load the dependencies of backends the app does not use."""

import json
import subprocess
import sys

import pytest


def _loaded_after(code: str, modules: list[str]) -> dict[str, bool]:
    """Run code in a fresh interpreter and report which of the given modules it left in sys.modules."""
    probe = f"{code}\nimport json, sys\nprint(json.dumps({{m: m in sys.modules for m in {modules!r}}}))"
    # Popen, not run: the autouse mock_pw_dump fixture of this folder replaces subprocess.run
    with subprocess.Popen([sys.executable, "-c", probe], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) as process:
        stdout, stderr = process.communicate(timeout=120)
    assert process.returncode == 0, stderr
    return json.loads(stdout.strip().splitlines()[-1])


@pytest.mark.parametrize(
    "code", ["import arduino.app_peripherals.microphone", "from arduino.app_peripherals.microphone import Microphone, BaseMicrophone"]
)
def test_import_does_not_load_the_backends_dependencies(code: str):
    heavy = ["arduino.app_peripherals.microphone.websocket_microphone", "websockets", "asyncio", "cryptography", "requests", "cv2", "PIL", "yaml"]
    assert _loaded_after(code, heavy) == dict.fromkeys(heavy, False)


def test_backends_resolve_to_their_submodule_classes():
    import arduino.app_peripherals.microphone as microphone
    from arduino.app_peripherals.microphone import alsa_microphone, websocket_microphone

    assert microphone.ALSAMicrophone is alsa_microphone.ALSAMicrophone
    assert microphone.WebSocketMicrophone is websocket_microphone.WebSocketMicrophone


def test_unknown_attribute_raises_attribute_error():
    import arduino.app_peripherals.microphone as microphone

    with pytest.raises(AttributeError):
        _ = microphone.DoesNotExist  # pyright: ignore[reportAttributeAccessIssue]
