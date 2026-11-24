# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""Runs the app in the forked child the way `python main.py` would: folder, environment, signals, `__main__`."""

import builtins
import importlib.machinery
import io
import os
import sys
import types
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum

from . import dependencies

STREAMLIT_PORT = 7000
"""The port Streamlit serves, the one run.sh always gave it."""


class Runner(StrEnum):
    PYTHON = "python"
    STREAMLIT = "streamlit"


@dataclass(frozen=True)
class Launch:
    """Everything the child needs to become the app."""

    app: str
    env: Mapping[str, str]
    """The container environment updated by the app one."""
    runner: Runner = Runner.PYTHON

    @property
    def main_py(self) -> str:
        return os.path.join(self.app, "python", "main.py")


def runner(app: str) -> Runner:
    """Returns how the app runs: through Streamlit when its app.yaml uses the streamlit_ui brick, as run.sh decides."""
    return Runner.STREAMLIT if dependencies.uses_streamlit(app) else Runner.PYTHON


def in_venv(venv: str) -> bool:
    """Tells whether this interpreter runs in the venv: a venv python puts its folder in sys.prefix."""
    return sys.prefix != sys.base_prefix and os.path.realpath(sys.prefix) == os.path.realpath(venv)


def environment(launch: Launch) -> dict[str, str]:
    """Builds the app environment: the launch one and the additions run.sh makes before running an app."""
    env = {**launch.env, "PYTHONUNBUFFERED": "1"}
    bricks = os.path.join(launch.app, "bricks")
    paths = [path for path in env.get("PYTHONPATH", "").split(os.pathsep) if path]
    if os.path.isdir(bricks) and bricks not in paths:
        env["PYTHONPATH"] = os.pathsep.join([*paths, bricks])
    # The app venv, when run.sh started the launcher in it
    venv = dependencies.venv_dir(launch.app)
    if in_venv(venv):
        env["VIRTUAL_ENV"] = venv
        env["PATH"] = os.pathsep.join(filter(None, [os.path.join(venv, "bin"), env.get("PATH")]))
    return env


def detach() -> None:
    """Makes the freshly forked process a group leader holding only the container stdio, first thing after the fork."""
    os.setsid()
    # Inherited fds are pointed at /dev/null, not closed: the launcher objects still own their numbers
    devnull = os.open(os.devnull, os.O_RDWR)
    for fd in _open_fds():
        if fd > 2 and fd != devnull:
            os.dup2(devnull, fd, inheritable=False)
    os.close(devnull)
    for stream in (sys.stdout, sys.stderr):
        if isinstance(stream, io.TextIOWrapper):
            stream.reconfigure(line_buffering=True)


def run(launch: Launch) -> int:
    """Turns the detached child into the app and runs it; returns its exit code.

    Raises:
        SystemExit: when the app exits through sys.exit(), as it would in its own interpreter.
    """
    reseed()
    os.chdir(launch.app)
    env = environment(launch)
    os.environ.clear()
    os.environ.update(env)
    sys.argv = [launch.main_py]
    # As a fresh interpreter: the script folder first, then PYTHONPATH
    sys.path[0] = os.path.dirname(launch.main_py)
    sys.path[1:1] = [path for path in env.get("PYTHONPATH", "").split(os.pathsep) if path and path not in sys.path]
    if launch.runner is Runner.STREAMLIT:
        return _run_streamlit(launch.main_py)
    _install_library_signal_handler()
    return _exec_main(launch.main_py)


def _run_streamlit(main_py: str) -> int:
    """Serves main.py with the Streamlit of this interpreter, as `streamlit run --server.port 7000 main.py` does."""
    from streamlit.web import bootstrap  # noqa: PLC0415 # pyright: ignore[reportMissingImports, reportUnknownVariableType]

    flags = {"server.port": STREAMLIT_PORT}
    bootstrap.load_config_options(flag_options=flags)  # pyright: ignore[reportUnknownMemberType]
    bootstrap.run(main_py, False, [], flags)  # pyright: ignore[reportUnknownMemberType]
    return 0


def reseed() -> None:
    """Reseeds what the warm interpreter seeded once and fork copied: NumPy's global generator; Python's random reseeds itself."""
    if (numpy := sys.modules.get("numpy")) is not None:
        numpy.random.seed()


def _install_library_signal_handler() -> None:
    """Installs the SIGTERM handler the library installs when main.py imports it, so that a stop before App.run() exits cleanly.

    A fresh interpreter gets it from the import; a warm one imported the library under the launcher's handlers, which kept
    the import from installing it, so the child, back on the default dispositions, asks for it again.
    """
    installer = getattr(sys.modules.get("arduino.app_utils.app"), "install_startup_signal_handler", None)
    if installer is not None:
        installer()


def _open_fds() -> list[int]:
    listing = "/proc/self/fd" if os.path.isdir("/proc/self/fd") else "/dev/fd"
    fds: list[int] = []
    for name in os.listdir(listing):
        try:
            os.fstat(int(name))
        except OSError:
            continue
        fds.append(int(name))
    return fds


def _exec_main(main_py: str) -> int:
    # A permanent module, not runpy: the app's threads and atexit handlers must keep seeing it as __main__
    module = types.ModuleType("__main__")
    module.__file__ = main_py
    module.__loader__ = importlib.machinery.SourceFileLoader("__main__", main_py)
    module.__spec__ = None
    module.__package__ = None
    module.__dict__["__cached__"] = None
    module.__dict__["__builtins__"] = builtins
    sys.modules["__main__"] = module
    try:
        with open(main_py, "rb") as f:
            code = compile(f.read(), main_py, "exec")
        exec(code, module.__dict__)  # noqa: S102
    except (SystemExit, KeyboardInterrupt):
        raise
    except BaseException as exc:
        tb = exc.__traceback__
        while tb is not None and tb.tb_frame.f_code.co_filename == __file__:
            tb = tb.tb_next
        # The default hook prints exc.__traceback__ whatever traceback it gets
        sys.excepthook(type(exc), exc.with_traceback(tb), tb)
        return 1
    finally:
        # As the interpreter does for a script: atexit handlers no longer see them
        module.__dict__.pop("__file__", None)
        module.__dict__.pop("__cached__", None)
    return 0
