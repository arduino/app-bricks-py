# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

import os

from arduino.app_tools.app_launcher import boot


def main() -> int:
    """Starts the app, then loads the launcher that takes it over."""
    started = boot.start_app(os.environ)
    from arduino.app_tools.app_launcher import launcher  # noqa: PLC0415

    return launcher.main(started)


if __name__ == "__main__":
    raise SystemExit(main())
