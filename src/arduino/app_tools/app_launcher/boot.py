# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""Starts the app of the container before the launcher loads: the app never waits for the modules the launcher imports."""

import os
import signal
import sys
import time
from collections.abc import Mapping
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from types import FrameType
from typing import Any, NoReturn

from . import child
from .child import Launch
from .config import Config


@dataclass(frozen=True)
class Started:
    """The app started before the launcher loaded, for the launcher to adopt."""

    launch: Launch
    pid: int
    started_at: datetime
    dispositions: Mapping[int, Any]
    """The handlers the stop signals had when the container started, the ones a fresh `python main.py` has."""


def start_app(env: Mapping[str, str], dispositions: Mapping[int, Any]) -> Started | None:
    """Forks the app the environment names and returns it; None when it cannot be started, for the launcher to report why.

    Args:
        env (Mapping[str, str]): The container environment.
        dispositions (Mapping[int, Any]): The handlers the stop signals had when the container started: the app gets them.

    Raises:
        SystemExit: in the forked child, with the exit code of the app.
    """
    try:
        config = Config.from_env(env)
    except ValueError:
        return None
    launch = Launch(app=config.app_dir, env={**env}, runner=child.runner(config.app_dir))
    if not os.path.isfile(launch.main_py):
        return None

    sys.stdout.flush()
    sys.stderr.flush()
    # A stop waits until the app has its own dispositions and the launcher knows which app to stop
    signal.pthread_sigmask(signal.SIG_BLOCK, dispositions)
    try:
        pid = os.fork()
    except OSError:
        signal.pthread_sigmask(signal.SIG_UNBLOCK, dispositions)
        return None
    if pid == 0:
        child.detach()
        restore(dispositions)
        signal.pthread_sigmask(signal.SIG_UNBLOCK, dispositions)
        raise SystemExit(child.run(launch))

    def stop(signum: int, frame: FrameType | None) -> NoReturn:
        _stop(pid, config.stop_timeout_s, signum)

    for signum in dispositions:
        signal.signal(signum, stop)
    signal.pthread_sigmask(signal.SIG_UNBLOCK, dispositions)
    return Started(launch=launch, pid=pid, started_at=datetime.now(UTC), dispositions=dispositions)


def restore(dispositions: Mapping[int, Any]) -> None:
    """Gives the stop signals the handlers they had when the container started."""
    for signum, handler in dispositions.items():
        signal.signal(signum, signal.SIG_DFL if handler is None else handler)


def _stop(pid: int, timeout_s: float, signum: int) -> NoReturn:
    """Stops the app while the launcher still loads, as the launcher does, and ends as an app stopped by the signal."""
    child.signal_group(pid, signal.SIGTERM)
    deadline = time.monotonic() + timeout_s
    with suppress(ChildProcessError):
        while os.waitpid(pid, os.WNOHANG) == (0, 0):
            if time.monotonic() >= deadline:
                child.signal_group(pid, signal.SIGKILL)
                os.waitpid(pid, 0)
                break
            time.sleep(0.01)
    raise SystemExit(128 + signum)
