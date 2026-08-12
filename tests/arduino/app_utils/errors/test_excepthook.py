# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

import sys

import arduino.app_utils.errors as errors
from arduino.app_utils.errors import AppError, _app_excepthook, install_excepthook


def _capture(exc: Exception):
    """Raises exc and returns its (type, value, traceback) as an excepthook would receive it."""
    try:
        raise exc
    except Exception:
        return sys.exc_info()


def test_app_error_prints_clean_report(capsys):
    exc_info = _capture(AppError("No speaker found at index 0", hint="Connect a speaker and restart the app."))
    _app_excepthook(*exc_info)

    err = capsys.readouterr().err
    assert "======== App failed to start " in err
    assert "Error in test_excepthook.py, line" in err
    assert "  No speaker found at index 0" in err
    assert "Hint: Connect a speaker and restart the app." in err
    assert "Trace:" in err
    assert "Traceback (most recent call last):" in err
    # The clean report comes before the traceback
    assert err.index("No speaker found") < err.index("Traceback")


def test_app_error_without_hint_omits_hint_line(capsys):
    exc_info = _capture(AppError("something went wrong"))
    _app_excepthook(*exc_info)

    err = capsys.readouterr().err
    assert "something went wrong" in err
    assert "Hint:" not in err


def test_app_error_subclass_is_reported(capsys):
    class FakePeripheralError(AppError):
        pass

    exc_info = _capture(FakePeripheralError("device is missing"))
    _app_excepthook(*exc_info)

    err = capsys.readouterr().err
    assert "======== App failed to start " in err
    assert "device is missing" in err


def test_unexpected_exceptions_are_reported_with_class_name(capsys):
    exc_info = _capture(ValueError("a user bug"))
    _app_excepthook(*exc_info)

    err = capsys.readouterr().err
    assert "======== App failed to start " in err
    assert "Unexpected error in test_excepthook.py, line" in err
    assert "  ValueError: a user bug" in err
    assert "Hint:" not in err
    assert "Trace:" in err
    assert "Traceback (most recent call last):" in err


def test_base_exceptions_are_delegated(monkeypatch):
    delegated = []
    monkeypatch.setattr(errors, "_previous_excepthook", lambda *args: delegated.append(args))

    try:
        raise KeyboardInterrupt()
    except KeyboardInterrupt:
        exc_info = sys.exc_info()
    _app_excepthook(*exc_info)

    assert delegated == [exc_info]


def test_install_excepthook_is_idempotent():
    saved_hook = sys.excepthook
    saved_previous = errors._previous_excepthook
    try:
        install_excepthook()
        install_excepthook()
        assert sys.excepthook is _app_excepthook
        # A double install must not chain the hook to itself
        assert errors._previous_excepthook is not _app_excepthook
    finally:
        sys.excepthook = saved_hook
        errors._previous_excepthook = saved_previous
