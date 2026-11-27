# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

import os
import signal


def _exit(signum: int, frame: object) -> None:
    raise SystemExit(128 + signum)


# First thing: the kernel ignores a signal left at its default action for PID 1, so a stop that lands while the modules
# below load would be lost. The app gets the handlers of before
dispositions: dict[int, object] = {signum: signal.signal(signum, _exit) for signum in (signal.SIGTERM, signal.SIGINT)}

from arduino.app_tools.app_launcher import boot  # noqa: E402


def main() -> int:
    """Starts the app, then loads the launcher that takes it over."""
    started = boot.start_app(os.environ, dispositions)
    if started is None:
        # The launcher reports why, and starts from the handlers it would have found
        boot.restore(dispositions)
    from arduino.app_tools.app_launcher import launcher  # noqa: PLC0415

    return launcher.main(started)


if __name__ == "__main__":
    raise SystemExit(main())
