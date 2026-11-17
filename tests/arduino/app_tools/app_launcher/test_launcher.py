# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
import venv
from pathlib import Path

import pytest

from arduino.app_tools.app_launcher import launcher as launcher_module

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="the launcher forks, POSIX only")

FAKE_MAIN = """
import os, signal, sys
print("hello from", __name__, sys.argv[0], os.environ["GREETING"], os.environ["INHERITED"], flush=True)
print("warning from", sys.argv[0], file=sys.stderr, flush=True)
signal.pause()
"""

# An app that leaves a helper behind in a session of its own, out of the app's process group, tagged by MARKER
LINGERING_MAIN = """
import os, signal, subprocess, sys
subprocess.Popen([sys.executable, "-c", "import signal; signal.pause()", os.environ["MARKER"]], start_new_session=True)
print("helper started", os.environ["MARKER"], flush=True)
signal.pause()
"""

# An app stopped while it builds its bricks, before App.run() takes the signals over
BUILDING_MAIN = """
import signal
import arduino.app_utils
print("building", flush=True)
signal.pause()
"""

# Stands in for streamlit.web.bootstrap, what the child runs a Streamlit app through
FAKE_BOOTSTRAP = """
import os, signal, sys
def load_config_options(flag_options):
    print("streamlit options", sorted(flag_options.items()), flush=True)
def run(main_script_path, is_hello, args, flag_options):
    print("streamlit serves", main_script_path, is_hello, args, os.getcwd(), flush=True)
    signal.pause()
"""

# Runs the launcher atop a pid namespace of its own, as in its container; unshare forwards no signal, it only waits
UNSHARE = ["unshare", "--user", "--map-root-user", "--pid", "--fork", "--mount-proc", "--kill-child"]


def can_unshare():
    if sys.platform != "linux" or shutil.which("unshare") is None:
        return False
    return subprocess.run([*UNSHARE, "true"], capture_output=True).returncode == 0


def wait_until(condition, timeout=5.0, details=lambda: ""):
    deadline = time.monotonic() + timeout
    while not (result := condition()):
        assert time.monotonic() < deadline, f"timed out {details()}"
        time.sleep(0.02)
    return result


def gone(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    return False


def tagged(marker):
    """The pids whose last argument is the marker, as seen from the test: from /proc on Linux, from ps elsewhere."""
    if not os.path.isdir("/proc"):
        listing = subprocess.run(["ps", "-A", "-ww", "-o", "pid=", "-o", "args="], capture_output=True, text=True, check=True).stdout
        return [int(pid) for pid, _, args in (line.strip().partition(" ") for line in listing.splitlines()) if args.endswith(f" {marker}")]
    return [pid for pid in proc_pids() if read_proc(pid, "cmdline").split("\0")[-2:-1] == [marker]]


def proc_pids():
    return [int(name) for name in os.listdir("/proc") if name.isdigit()]


def read_proc(pid, name):
    try:
        with open(f"/proc/{pid}/{name}", encoding="utf-8", errors="replace") as file:
            return file.read()
    except OSError:
        return ""


def write_app(app, app_yaml="", main=FAKE_MAIN, requirements=None):
    """Writes an app folder; without a main, one the launcher cannot start."""
    (app / "python").mkdir(parents=True, exist_ok=True)
    if main is not None:
        # In one step: an app that is starting reads the old file or the new one, never half of it
        staged = app / "python" / "main.py.new"
        staged.write_text(main)
        staged.replace(app / "python" / "main.py")
    (app / "app.yaml").write_text(app_yaml)
    if requirements is not None:
        (app / "python" / "requirements.txt").write_text(requirements)
    return app


def request(message, socket_path, timeout_s=10):
    """Sends one request to the launcher and returns its response."""
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
        sock.settimeout(timeout_s)
        sock.connect(socket_path)
        sock.sendall(json.dumps(message).encode() + b"\n")
        return json.loads(sock.makefile("rb").readline())


class LauncherProcess:
    """A launcher serving the app folder it creates, as run.sh starts one for /app."""

    def __init__(self, preload="json", wrapper=(), extra_modules=None, in_venv=False, env=None, **app):
        # AF_UNIX paths are limited to about 100 bytes
        self.root = Path(tempfile.mkdtemp(prefix="launcher", dir="/tmp")).resolve()
        self.socket = str(self.root / "launcher.sock")
        self.served = write_app(self.root / "app", **app)
        self.venv = self.served / ".cache" / ".venv"
        interpreter = sys.executable
        if in_venv:
            # As run.sh starts the launcher for an app with dependencies: on the interpreter of the app venv
            venv.EnvBuilder(system_site_packages=True, symlinks=True).create(self.venv)
            interpreter = str(self.venv / "bin" / "python")
        modules = self.root / "modules"
        (modules / "streamlit" / "web").mkdir(parents=True)
        (modules / "streamlit" / "__init__.py").touch()
        (modules / "streamlit" / "web" / "__init__.py").touch()
        (modules / "streamlit" / "web" / "bootstrap.py").write_text(FAKE_BOOTSTRAP)
        for name, source in (extra_modules or {}).items():
            (modules / name).write_text(source)
        # A venv of the test, unlike one of the board, does not see the library through its base interpreter
        library = str(Path(launcher_module.__file__).resolve().parents[3]) if in_venv else None
        launcher_env = {
            **os.environ,
            "PYTHONPATH": os.pathsep.join(filter(None, [str(modules), library, os.environ.get("PYTHONPATH")])),
            "APP_DIR": str(self.served),
            "APP_LAUNCHER_PRELOAD": preload,
            "APP_LAUNCHER_SOCKET": self.socket,
            "APP_LAUNCHER_STOP_TIMEOUT_S": "1",
            "APP_LAUNCHER_LOG_LEVEL": "info",
            "GREETING": "from the container",
            "INHERITED": "yes",
            **(env or {}),
        }
        self.wrapped = bool(wrapper)
        with open(self.root / "stdout.log", "wb") as stdout, open(self.root / "stderr.log", "wb") as stderr:
            self.process = subprocess.Popen(
                [*wrapper, interpreter, "-m", "arduino.app_tools.app_launcher"], env=launcher_env, stdout=stdout, stderr=stderr
            )

    def wait_ready(self):
        wait_until(self.answers, details=lambda: (self.root / "stderr.log").read_text())
        return self

    def answers(self):
        try:
            return self.request({"cmd": "status"})["ok"]
        except OSError:
            return False

    def request(self, message):
        return request(message, self.socket)

    def start(self, **fields):
        """Asks the launcher to run its app again."""
        return self.request({"cmd": "start", **fields})

    def output(self, stream="stdout", until=""):
        """Waits for what the launcher and its apps wrote to the stream to hold `until` and returns it."""
        log = self.root / f"{stream}.log"
        return wait_until(lambda: until in (text := log.read_text()) and text, details=lambda: log.read_text())

    def app(self, **fields):
        """Writes the served app again, as an edit of it does."""
        return write_app(self.served, **fields)

    def child(self):
        return self.request({"cmd": "status"})["child"]

    def running(self):
        """Waits for the launcher to answer and returns the app it runs."""
        return wait_until(self.wait_ready().child, details=lambda: (self.root / "stderr.log").read_text())

    def close(self):
        if self.process.poll() is None:
            os.kill(self.launcher_pid(), signal.SIGTERM)
        return self.process.wait(5)

    def launcher_pid(self):
        """The launcher process as seen from the test: the child of the namespace wrapper, when there is one."""
        if not self.wrapped:
            return self.process.pid
        return next(pid for pid in proc_pids() if f"\nPPid:\t{self.process.pid}\n" in read_proc(pid, "status"))

    def remove(self):
        self.close()
        shutil.rmtree(self.root)


@pytest.fixture(scope="module")
def launcher():
    launcher = LauncherProcess().wait_ready()
    yield launcher
    launcher.remove()


def test_runs_its_app_at_once_until_a_start_replaces_it(launcher):
    app = launcher.served
    started = launcher.running()
    status = launcher.request({"cmd": "status"})
    assert status["child"] == {"app": str(app), "pid": started["pid"], "runner": "python", "started_at": started["started_at"], "stopping": False}
    assert status["preload"]["modules"] == ["json"] and status["preload"]["failed"] == []
    # The preload runs once the launcher is idle
    wait_until(lambda: launcher.request({"cmd": "status"})["preload"]["loaded"] == ["json"], details=lambda: launcher.output("stderr"))
    assert set(status) == {"ok", "child", "last", "preload", "version", "uptime_s", "venv"}

    # The app writes to the container stdout and stderr, as the launcher does
    assert f"hello from __main__ {app}/python/main.py from the container yes\n" in launcher.output(until=f"{app}/python/main.py")
    assert f"warning from {app}/python/main.py\n" in launcher.output("stderr", until=f"warning from {app}")
    assert "app-launcher: Ready on" in launcher.output("stderr")

    replaced = launcher.start()
    assert replaced["ok"] and replaced["runner"] == "python" and replaced["pid"] != started["pid"]
    assert replaced["timings"]["stop_ms"] > 0 and set(replaced["timings"]) == {"stop_ms", "fork_ms"}
    assert gone(started["pid"])
    status = launcher.request({"cmd": "status"})
    assert status["child"]["pid"] == replaced["pid"] and status["child"]["started_at"] == replaced["started_at"]
    assert launcher.process.poll() is None, "the launcher stays when it replaces its app"
    assert status["last"] == {
        "app": str(app),
        "pid": started["pid"],
        "exit_code": None,
        "signal": signal.SIGTERM,
        "ended_at": status["last"]["ended_at"],
    }


def test_a_start_runs_the_app_as_edited(launcher):
    app = launcher.app(main='print("edited", flush=True)\nimport signal\nsignal.pause()\n')
    assert launcher.start()["ok"]
    assert "edited\n" in launcher.output(until="edited")
    assert launcher.child()["app"] == str(app)


def test_the_request_env_is_laid_over_the_container_one(launcher):
    app = launcher.app()
    started = launcher.start(env={"GREETING": "hi"})
    assert started["ok"]
    assert f"hello from __main__ {app}/python/main.py hi yes\n" in launcher.output(until="main.py hi yes")


def test_an_app_stopped_before_app_run_exits_cleanly():
    # The library installs its startup SIGTERM handler when imported: a warm interpreter imported it already
    launcher = LauncherProcess(preload="arduino.app_utils", main=BUILDING_MAIN).wait_ready()
    try:
        wait_until(lambda: launcher.request({"cmd": "status"})["preload"]["loaded"] == ["arduino.app_utils"], timeout=30)
        # The app the launcher started at once imported the library itself: this one inherits it
        started = launcher.start()
        assert started["ok"], started
        wait_until(lambda: launcher.output().count("building") == 2)
        assert launcher.start()["ok"]
        last = launcher.request({"cmd": "status"})["last"]
        assert (last["pid"], last["exit_code"], last["signal"]) == (started["pid"], 143, None)
    finally:
        launcher.remove()


@pytest.mark.skipif(not can_unshare(), reason="needs Linux and unprivileged user and pid namespaces")
def test_what_a_stopped_app_left_behind_is_killed():
    marker = f"lingering-{os.getpid()}-{time.monotonic_ns()}"
    contained = LauncherProcess(wrapper=UNSHARE, main=LINGERING_MAIN, env={"MARKER": marker})
    try:
        started = contained.running()
        contained.output(until=f"helper started {marker}")
        assert tagged(marker), "the helper runs"

        contained.app()
        assert contained.start()["ok"]
        assert gone(started["pid"])
        assert wait_until(lambda: not tagged(marker)), "the helper, in a session of its own, outlived the app"
        assert "Killed what was left behind" in contained.output("stderr")
        assert contained.answers(), "the launcher survives its own sweep"
    finally:
        contained.remove()


def test_serves_a_streamlit_app_through_streamlit_on_port_7000():
    launcher = LauncherProcess(app_yaml="bricks:\n  - arduino:streamlit_ui: {}\n")
    try:
        assert launcher.running()["runner"] == "streamlit"
        main_py = launcher.served / "python" / "main.py"
        assert f"streamlit serves {main_py} False [] {launcher.served}\n" in launcher.output(until="streamlit serves")
        assert "streamlit options [('server.port', 7000)]" in launcher.output()
        assert launcher.start()["ok"]
        assert launcher.request({"cmd": "status"})["last"]["signal"] == signal.SIGTERM
    finally:
        launcher.remove()


def test_an_app_whose_dependencies_changed_is_stale():
    launcher = LauncherProcess(requirements="requests\n")
    try:
        started = launcher.running()

        # Dependencies run.sh did not install at the container start: the app needs a fresh container
        requirements = launcher.served / "python" / "requirements.txt"
        requirements.write_text("requests==2.32.0\n")
        refused = launcher.start()
        assert refused["error"]["code"] == "stale"
        assert launcher.child()["pid"] == started["pid"], "the running app goes on"

        requirements.write_text("requests\n")
        assert launcher.start()["ok"]
    finally:
        launcher.remove()


VENV_MAIN = """
import os, signal, sys
print("prefix", sys.prefix, "venv", os.environ.get("VIRTUAL_ENV"), flush=True)
signal.pause()
"""


def test_an_app_with_a_venv_runs_in_it_with_its_streamlit():
    launcher = LauncherProcess(in_venv=True, main=VENV_MAIN, requirements="requests\n").wait_ready()
    try:
        status = launcher.request({"cmd": "status"})
        assert status["venv"] == str(launcher.venv)
        assert f"prefix {launcher.venv} venv {launcher.venv}\n" in launcher.output(until="prefix")
        wait_until(lambda: launcher.request({"cmd": "status"})["preload"]["loaded"] == ["json", "streamlit.web.bootstrap"])
    finally:
        launcher.remove()


def test_refuses_what_it_cannot_start(launcher):
    launcher.app()
    running = launcher.start(env={"GREETING": "refusals"})["pid"]
    # Loaded before its file goes: an app that cannot read its main.py ends, and its launcher with it
    launcher.output(until="main.py refusals yes")
    assert launcher.start(env={"GREETING": 1})["error"]["code"] == "bad_request"
    (launcher.served / "python" / "main.py").unlink()
    assert launcher.start()["error"]["code"] == "not_found"
    assert launcher.child()["pid"] == running, "a refused start changes nothing"


# Last: it stops the shared launcher
def test_exits_on_sigterm_stopping_its_app(launcher):
    launcher.app()
    started = launcher.start()
    assert launcher.close() == 128 + signal.SIGTERM, "the launcher ends as an app stopped by a signal does"
    assert gone(started["pid"])
    assert not os.path.exists(launcher.socket)


def test_without_an_app_folder_it_refuses_to_run():
    env = {name: value for name, value in os.environ.items() if name != "APP_DIR"}
    refused = subprocess.run([sys.executable, "-m", "arduino.app_tools.app_launcher"], env=env, capture_output=True, text=True, timeout=30)
    assert refused.returncode == 2
    assert "APP_DIR" in refused.stderr


@pytest.mark.parametrize(
    ("main", "exit_code"),
    [("raise SystemExit(3)\n", 3), ("", 0), ("import os, signal\nos.kill(os.getpid(), signal.SIGKILL)\n", 128 + signal.SIGKILL)],
    ids=["failed", "returned", "killed"],
)
def test_ends_as_its_app_did(main, exit_code):
    launcher = LauncherProcess(main=main)
    try:
        assert launcher.process.wait(10) == exit_code, "the container tells how its app ended, as one that ran the app alone"
        assert not os.path.exists(launcher.socket)
    finally:
        launcher.remove()


def test_ends_as_failed_when_a_start_leaves_no_app_running():
    # A thread the preload leaves alive makes the launcher refuse to fork
    threaded = "import threading, time\nthreading.Thread(target=time.sleep, args=(60,), daemon=True).start()\n"
    launcher = LauncherProcess(preload="threaded", extra_modules={"threaded.py": threaded})
    try:
        launcher.running()
        wait_until(lambda: launcher.request({"cmd": "status"})["preload"]["loaded"] == ["threaded"])
        refused = launcher.start()
        assert refused["error"]["code"] == "internal", "the running app was stopped, the new one could not be forked"
        assert launcher.process.wait(10) == 1
    finally:
        launcher.remove()


def test_ends_as_failed_when_it_cannot_start_its_app():
    launcher = LauncherProcess(main=None)
    try:
        assert launcher.process.wait(10) == 1
        assert "Cannot start" in (launcher.root / "stderr.log").read_text()
    finally:
        launcher.remove()


# An app that ends once its launcher is gone: in a container it dies with the launcher, here nothing else ends it
ORPHANED_MAIN = """
import os, time
launcher = os.getppid()
print("running", flush=True)
while os.getppid() == launcher:
    time.sleep(0.05)
"""
SLOW_PRELOAD = {"slow.py": "import time\nprint('importing', flush=True)\ntime.sleep(30)\n"}


def test_the_app_does_not_wait_for_the_preload():
    launcher = LauncherProcess(preload="slow", extra_modules=SLOW_PRELOAD, main=ORPHANED_MAIN)
    try:
        launcher.output(until="running")
        assert "importing" not in launcher.output(), "the app starts before the preload"
    finally:
        launcher.remove()


def test_exits_on_sigterm_during_the_preload():
    launcher = LauncherProcess(preload="slow", extra_modules=SLOW_PRELOAD, main=ORPHANED_MAIN)
    try:
        launcher.output(until="importing")
        launcher.process.send_signal(signal.SIGTERM)
        assert launcher.process.wait(5) == 128 + signal.SIGTERM
    finally:
        launcher.remove()


def test_app_bricks_modules_follow_the_app_descriptor(tmp_path):
    (tmp_path / "app.yaml").write_text(
        "name: demo\nbricks:\n  - arduino:llm:\n      model: llamacpp:Qwen3.5-0.8B-Q4_0\n  - arduino:video_object_detection\n"
        "  - arduino:web_ui\n  - acme:custom: {}\n  - arduino:llm\n"
    )
    from arduino.app_tools.app_launcher.bricks import app_bricks_modules

    # The package of a brick is read from its brick_config.yaml: video_object_detection lives in video_objectdetection
    assert app_bricks_modules(str(tmp_path)) == [
        "arduino.app_bricks.llm",
        "arduino.app_bricks.video_objectdetection",
        "arduino.app_bricks.web_ui",
    ]
    assert app_bricks_modules(str(tmp_path / "missing")) == []
