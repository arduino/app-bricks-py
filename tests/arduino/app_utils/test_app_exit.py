# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""How an app process ends on SIGTERM: promptly, with exit code 143, and running its exit handlers, whether it is still loading or running."""

import signal
import subprocess
import sys
import textwrap
import time

import pytest

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="SIGTERM delivery is a POSIX behaviour")


def _spawn(code: str) -> subprocess.Popen[str]:
    return subprocess.Popen([sys.executable, "-u", "-c", textwrap.dedent(code)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)


def _wait_for(proc: subprocess.Popen[str], marker: str, timeout: float = 60.0) -> str:
    """Read the child's output until marker shows up, returning what was read."""
    assert proc.stdout is not None
    deadline = time.monotonic() + timeout
    lines = ""
    while time.monotonic() < deadline:
        line = proc.stdout.readline()
        lines += line
        if marker in line:
            return lines
        if line == "" and proc.poll() is not None:
            break
    raise AssertionError(f"{marker!r} not seen in:\n{lines}")


def test_sigterm_while_the_app_is_still_loading_ends_the_process():
    """Before App.run() nothing handles SIGTERM: as PID 1 of a container the process would ignore it."""
    proc = _spawn("""
        import atexit, time
        import arduino.app_utils
        atexit.register(lambda: print("ATEXIT RAN", flush=True))
        print("LOADING", flush=True)
        time.sleep(60)  # A slow import
    """)
    _wait_for(proc, "LOADING")
    t = time.monotonic()
    proc.send_signal(signal.SIGTERM)
    out = proc.communicate(timeout=30)[0]
    assert time.monotonic() - t < 10, "the process did not end promptly on SIGTERM"
    assert proc.returncode == 128 + signal.SIGTERM
    assert "ATEXIT RAN" in out


def test_sigterm_while_running_ends_the_process_right_after_the_shutdown():
    """The process ends as soon as the shutdown is done: no interpreter teardown, no wait on leftover threads."""
    proc = _spawn("""
        import atexit, threading, time
        from arduino.app_utils import App
        atexit.register(lambda: print("ATEXIT RAN", flush=True))
        threading.Thread(target=time.sleep, args=(120,), name="leftover").start()  # Non-daemon: sys.exit() would wait for it
        App.run()
        print("AFTER RUN", flush=True)  # Not reached: run() exits on a signal
    """)
    _wait_for(proc, "App started")
    time.sleep(1)  # Past the handler installation in loop(): this is the signal path, not the startup one
    t = time.monotonic()
    proc.send_signal(signal.SIGTERM)
    out = proc.communicate(timeout=30)[0]
    assert time.monotonic() - t < 15, "the process waited past the shutdown"
    assert proc.returncode == 128 + signal.SIGTERM
    assert "App shutdown completed" in out
    assert "ATEXIT RAN" in out, "exit handlers must still run"
    assert "AFTER RUN" not in out


def test_a_user_loop_that_stops_returns_from_run_as_before():
    """The fast exit is for the signal path only: a loop that ends by itself hands control back to the script."""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            textwrap.dedent("""
                from arduino.app_utils import App
                def loop():
                    raise StopIteration
                App.run(loop)
                print("AFTER RUN", flush=True)
            """),
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert "AFTER RUN" in result.stdout


def test_startup_handler_leaves_an_existing_handler_alone():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            textwrap.dedent("""
                import signal
                mine = lambda signum, frame: None
                signal.signal(signal.SIGTERM, mine)
                import arduino.app_utils
                assert signal.getsignal(signal.SIGTERM) is mine
            """),
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr


def test_startup_handler_is_installed_by_the_import():
    result = subprocess.run(
        [sys.executable, "-c", "import signal, arduino.app_utils; assert signal.getsignal(signal.SIGTERM) is not signal.SIG_DFL"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
