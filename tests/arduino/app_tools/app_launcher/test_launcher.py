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
    (app / "python").mkdir(parents=True, exist_ok=True)
    (app / "python" / "main.py").write_text(main)
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
    def __init__(self, preload="json", wrapper=(), extra_modules=None, served=None, interpreter=sys.executable):
        # AF_UNIX paths are limited to about 100 bytes
        self.root = Path(tempfile.mkdtemp(prefix="launcher", dir="/tmp")).resolve()
        self.socket = str(self.root / "launcher.sock")
        self.apps = self.root / "apps"
        modules = self.root / "modules"
        (modules / "streamlit" / "web").mkdir(parents=True)
        (modules / "streamlit" / "__init__.py").touch()
        (modules / "streamlit" / "web" / "__init__.py").touch()
        (modules / "streamlit" / "web" / "bootstrap.py").write_text(FAKE_BOOTSTRAP)
        for name, source in (extra_modules or {}).items():
            (modules / name).write_text(source)
        # A venv of the test, unlike one of the board, does not see the library through its base interpreter
        library = str(Path(launcher_module.__file__).resolve().parents[3]) if interpreter != sys.executable else None
        env = {
            **os.environ,
            "PYTHONPATH": os.pathsep.join(filter(None, [str(modules), library, os.environ.get("PYTHONPATH")])),
            "APP_LAUNCHER_PRELOAD": preload,
            "APP_LAUNCHER_SOCKET": self.socket,
            "APP_LAUNCHER_STOP_TIMEOUT_S": "1",
            "APP_LAUNCHER_LOG_LEVEL": "info",
            "GREETING": "from the container",
            "INHERITED": "yes",
        }
        env.pop("APP_DIR", None)
        if served is not None:
            env["APP_DIR"] = str(served)
        self.wrapped = bool(wrapper)
        with open(self.root / "stdout.log", "wb") as stdout, open(self.root / "stderr.log", "wb") as stderr:
            self.process = subprocess.Popen([*wrapper, interpreter, "-m", "arduino.app_tools.app_launcher"], env=env, stdout=stdout, stderr=stderr)

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

    def output(self, stream="stdout", until=""):
        """Waits for what the launcher and its apps wrote to the stream to hold `until` and returns it."""
        log = self.root / f"{stream}.log"
        return wait_until(lambda: until in (text := log.read_text()) and text, details=lambda: log.read_text())

    def app(self, name, **fields):
        return write_app(self.apps / name, **fields)

    def child(self):
        return self.request({"cmd": "status"})["child"]

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


def test_runs_an_app_until_stopped(launcher):
    app = launcher.app("hello")
    started = launcher.request({"cmd": "start", "app": str(app)})
    assert started["ok"] and started["runner"] == "python"
    assert set(started["timings"]) == {"stop_ms", "fork_ms"}

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

    stopped = launcher.request({"cmd": "stop"})
    assert stopped == {"ok": True, "stopped": [{"app": str(app), "pid": started["pid"], "exit_code": None, "signal": signal.SIGTERM}]}
    assert gone(started["pid"])
    status = launcher.request({"cmd": "status"})
    assert status["child"] is None
    assert status["last"] == {
        "app": str(app),
        "pid": started["pid"],
        "exit_code": None,
        "signal": signal.SIGTERM,
        "ended_at": status["last"]["ended_at"],
    }
    assert launcher.request({"cmd": "stop"}) == {"ok": True, "stopped": []}


def test_the_request_env_is_laid_over_the_container_one(launcher):
    app = launcher.app("greeting")
    started = launcher.request({"cmd": "start", "app": str(app), "env": {"GREETING": "hi"}})
    assert started["ok"]
    assert f"hello from __main__ {app}/python/main.py hi yes\n" in launcher.output(until=f"{app}/python/main.py")
    launcher.request({"cmd": "stop"})


def test_a_start_stops_the_running_app_first(launcher):
    app = launcher.app("again")
    first = launcher.request({"cmd": "start", "app": str(app)})
    second = launcher.request({"cmd": "start", "app": str(app)})
    assert second["ok"] and second["pid"] != first["pid"] and second["timings"]["stop_ms"] > 0
    assert gone(first["pid"])
    assert launcher.child()["pid"] == second["pid"]

    third = launcher.request({"cmd": "start", "app": str(launcher.app("other"))})
    assert third["ok"] and gone(second["pid"])
    assert launcher.request({"cmd": "stop"})["stopped"][0]["pid"] == third["pid"]


def test_an_app_dying_on_its_own_is_remembered_as_the_last_one(launcher):
    app = launcher.app("crashing", main="raise SystemExit(3)\n")
    started = launcher.request({"cmd": "start", "app": str(app)})
    last = wait_until(lambda: launcher.request({"cmd": "status"})["last"] if launcher.child() is None else None)
    assert (last["app"], last["pid"], last["exit_code"], last["signal"]) == (str(app), started["pid"], 3, None)


def test_a_stop_before_app_run_exits_the_app_cleanly():
    # The library installs its startup SIGTERM handler when imported: a warm interpreter imported it already
    launcher = LauncherProcess(preload="arduino.app_utils").wait_ready()
    try:
        wait_until(lambda: launcher.request({"cmd": "status"})["preload"]["loaded"] == ["arduino.app_utils"], timeout=30)
        app = launcher.app("building", main=BUILDING_MAIN)
        started = launcher.request({"cmd": "start", "app": str(app)})
        assert started["ok"], started
        launcher.output(until="building")
        stopped = launcher.request({"cmd": "stop"})
        assert stopped["stopped"] == [{"app": str(app), "pid": started["pid"], "exit_code": 143, "signal": None}]
    finally:
        launcher.remove()


@pytest.mark.skipif(not can_unshare(), reason="needs Linux and unprivileged user and pid namespaces")
def test_a_stop_kills_what_the_app_left_behind():
    contained = LauncherProcess(wrapper=UNSHARE).wait_ready()
    try:
        marker = f"lingering-{contained.root.name}"
        app = contained.app("lingering", main=LINGERING_MAIN)
        started = contained.request({"cmd": "start", "app": str(app), "env": {"MARKER": marker}})
        contained.output(until=f"helper started {marker}")
        assert tagged(marker), "the helper runs"

        stopped = contained.request({"cmd": "stop"})
        assert stopped["stopped"][0]["pid"] == started["pid"]
        assert wait_until(lambda: not tagged(marker)), "the helper, in a session of its own, outlived the app"
        assert "Killed what was left behind" in contained.output("stderr")
        assert contained.answers(), "the launcher survives its own sweep"
    finally:
        contained.remove()


def test_serves_streamlit_apps_through_streamlit_on_port_7000(launcher):
    app = launcher.app("dashboard", app_yaml="bricks:\n  - arduino:streamlit_ui: {}\n")
    started = launcher.request({"cmd": "start", "app": str(app)})
    assert started["runner"] == "streamlit"
    main_py = app / "python" / "main.py"
    assert f"streamlit serves {main_py} False [] {app}\n" in launcher.output(until="streamlit serves")
    assert "streamlit options [('server.port', 7000)]" in launcher.output()
    assert launcher.request({"cmd": "stop"})["stopped"][0]["signal"] == signal.SIGTERM


def test_a_served_folder_whose_dependencies_changed_is_stale():
    served = write_app(Path(tempfile.mkdtemp(prefix="served", dir="/tmp")).resolve(), requirements="requests\n")
    launcher = LauncherProcess(served=served).wait_ready()
    try:
        started = launcher.request({"cmd": "start", "app": str(served)})
        assert started["ok"], started
        other = launcher.app("other")
        assert launcher.request({"cmd": "start", "app": str(other)})["error"]["code"] == "bad_request"

        # Dependencies run.sh did not install at the container start: the app needs a fresh container
        (served / "python" / "requirements.txt").write_text("requests==2.32.0\n")
        refused = launcher.request({"cmd": "start", "app": str(served)})
        assert refused["error"]["code"] == "stale"
        assert launcher.child()["pid"] == started["pid"], "the running app goes on"

        (served / "python" / "requirements.txt").write_text("requests\n")
        assert launcher.request({"cmd": "start", "app": str(served)})["ok"]
        assert launcher.request({"cmd": "stop"})["ok"]
    finally:
        launcher.remove()
        shutil.rmtree(served)


VENV_MAIN = """
import os, signal, sys
print("prefix", sys.prefix, "venv", os.environ.get("VIRTUAL_ENV"), flush=True)
signal.pause()
"""


def test_a_served_app_with_a_venv_runs_in_it_with_its_streamlit():
    # As run.sh serve starts the launcher for an app with dependencies: on the interpreter of the app venv
    served = write_app(Path(tempfile.mkdtemp(prefix="served", dir="/tmp")).resolve(), main=VENV_MAIN, requirements="requests\n")
    venv_dir = served / ".cache" / ".venv"
    venv.EnvBuilder(system_site_packages=True, symlinks=True).create(venv_dir)
    launcher = LauncherProcess(served=served, interpreter=str(venv_dir / "bin" / "python")).wait_ready()
    try:
        status = launcher.request({"cmd": "status"})
        assert status["venv"] == str(venv_dir)
        started = launcher.request({"cmd": "start", "app": str(served)})
        assert started["ok"], started
        assert f"prefix {venv_dir} venv {venv_dir}\n" in launcher.output(until="prefix")
        wait_until(lambda: launcher.request({"cmd": "status"})["preload"]["loaded"] == ["json", "streamlit.web.bootstrap"])
        assert launcher.request({"cmd": "stop"})["ok"]
    finally:
        launcher.remove()
        shutil.rmtree(served)


def test_without_a_served_folder_the_first_start_fixes_the_dependencies(launcher):
    app = launcher.app("deps", requirements="requests\n")
    assert launcher.request({"cmd": "start", "app": str(app)})["ok"]
    (app / "python" / "requirements.txt").write_text("requests==2.32.0\n")
    assert launcher.request({"cmd": "start", "app": str(app)})["error"]["code"] == "stale"
    assert launcher.request({"cmd": "stop"})["ok"]


def test_refuses_what_is_not_an_app(launcher):
    assert launcher.request({"cmd": "start", "app": "relative/app"})["error"]["code"] == "bad_request"
    assert launcher.request({"cmd": "start", "app": str(launcher.apps / "missing")})["error"]["code"] == "not_found"
    (launcher.apps / "empty").mkdir()
    assert launcher.request({"cmd": "start", "app": str(launcher.apps / "empty")})["error"]["code"] == "not_found"
    refused = launcher.request({"cmd": "start", "app": str(launcher.app("bad-env")), "env": {"GREETING": 1}})
    assert refused["error"]["code"] == "bad_request"
    assert launcher.child() is None


# Last: it stops the shared launcher
def test_exits_on_sigterm_stopping_its_app(launcher):
    started = launcher.request({"cmd": "start", "app": str(launcher.app("last"))})
    assert launcher.close() == 128 + signal.SIGTERM, "the launcher ends as an app stopped by a signal does"
    assert gone(started["pid"])
    assert not os.path.exists(launcher.socket)


def test_a_start_does_not_wait_for_the_preload():
    launcher = LauncherProcess(preload="slow", extra_modules={"slow.py": "import time\ntime.sleep(30)\n"}).wait_ready()
    try:
        app = launcher.app("early")
        started = launcher.request({"cmd": "start", "app": str(app)})
        assert started["ok"], started
        assert launcher.request({"cmd": "status"})["preload"]["loaded"] == []
        launcher.output(until="hello from __main__")
    finally:
        launcher.remove()


def test_exits_on_sigterm_during_the_preload():
    launcher = LauncherProcess(preload="slow", extra_modules={"slow.py": "import time\nprint('importing', flush=True)\ntime.sleep(30)\n"})
    try:
        launcher.output(until="importing")
        launcher.process.send_signal(signal.SIGTERM)
        assert launcher.process.wait(5) == 128 + signal.SIGTERM
    finally:
        launcher.remove()


def test_app_bricks_modules_follow_the_app_descriptor(tmp_path):
    (tmp_path / "app.yaml").write_text(
        "name: demo\nbricks:\n  - arduino:llm:\n      model: llamacpp:Qwen3.5-0.8B-Q4_0\n  - arduino:web_ui\n  - acme:custom: {}\n  - arduino:llm\n"
    )
    from arduino.app_tools.app_launcher.launcher import app_bricks_modules

    assert app_bricks_modules(str(tmp_path)) == ["arduino.app_bricks.llm", "arduino.app_bricks.web_ui"]
    assert app_bricks_modules(str(tmp_path / "missing")) == []
