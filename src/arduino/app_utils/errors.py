# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

import os
import sys
import traceback
from types import TracebackType
from typing import Any

__all__ = ["AppError"]

_BANNER_WIDTH = 53


class AppError(Exception):
    """Base class for user-facing application errors.

    These signal environment or configuration problems (missing hardware, invalid
    settings, unreachable services) rather than programming errors. When one of
    these reaches the top level uncaught, the app prints a concise, readable
    report explaining what went wrong, followed by the full traceback.

    Args:
        *args: Standard Exception arguments, typically the error message.
        hint (str, optional): A suggestion telling the user how to fix the problem.

    Examples:
        raise SpeakerOpenError("No speaker found", hint="Connect a speaker and restart the app.")
    """

    def __init__(self, *args: Any, hint: str | None = None) -> None:
        super().__init__(*args)
        self.hint = hint


def _banner(text: str = "") -> str:
    if text:
        return f"======== {text} ".ljust(_BANNER_WIDTH, "=")
    return "=" * _BANNER_WIDTH


def _find_user_frame(tb: TracebackType | None) -> traceback.FrameSummary | None:
    """Returns the innermost traceback frame that belongs to user code.

    Frames coming from installed packages (site-packages/dist-packages) or from
    synthetic sources ("<frozen ...>", "<string>") are skipped, so the reported
    location points at the user's own script whenever possible.
    """
    user_frames = [
        frame
        for frame in traceback.extract_tb(tb)
        if "site-packages" not in frame.filename and "dist-packages" not in frame.filename and not frame.filename.startswith("<")
    ]
    return user_frames[-1] if user_frames else None


def _app_excepthook(exc_type: type[BaseException], exc: BaseException, tb: TracebackType | None) -> None:
    if not isinstance(exc, Exception):
        # BaseExceptions like KeyboardInterrupt keep the standard behavior
        _previous_excepthook(exc_type, exc, tb)
        return

    out = sys.stderr
    frame = _find_user_frame(tb)
    location = f" in {os.path.basename(frame.filename)}, line {frame.lineno}" if frame else ""

    print(_banner("App failed to start"), file=out)
    if isinstance(exc, AppError):
        print(f"Error{location}:", file=out)
        print(f"  {str(exc) or exc_type.__name__}", file=out)
        if exc.hint:
            print(f"\nHint: {exc.hint}", file=out)
    else:
        detail = f"{exc_type.__name__}: {exc}" if str(exc) else exc_type.__name__
        print(f"Unexpected error{location}:", file=out)
        print(f"  {detail}", file=out)
    print("\nTrace:", file=out)
    traceback.print_exception(exc_type, exc, tb, file=out)
    print(_banner(), file=out, flush=True)


_previous_excepthook = sys.__excepthook__


def install_excepthook() -> None:
    """Installs the global app excepthook. Idempotent.

    Every uncaught Exception is reported with a clean, user-readable message
    before the full traceback. AppErrors are shown with their message and
    optional hint; any other exception is reported as unexpected, with its
    class name. BaseExceptions (e.g. KeyboardInterrupt) are handled by the
    previously installed hook.
    """
    global _previous_excepthook
    if sys.excepthook is _app_excepthook:
        return
    _previous_excepthook = sys.excepthook
    sys.excepthook = _app_excepthook
