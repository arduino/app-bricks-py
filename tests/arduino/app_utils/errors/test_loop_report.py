# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""How App.loop() ends when the user loop raises: errors get the readable report, StopIteration a clean stop."""

import signal

import pytest

from arduino.app_utils.app import AppController
from arduino.app_utils.errors import AppError


@pytest.fixture
def app():
    # loop() installs its own SIGTERM handler on the main thread: put the previous one back
    saved = signal.getsignal(signal.SIGTERM)
    try:
        yield AppController()
    finally:
        signal.signal(signal.SIGTERM, saved)


def test_unexpected_error_in_user_loop_is_reported(app, capsys):
    def user_loop():
        raise FileNotFoundError("Image file not found: assets/photo.jpg")

    assert app.loop(user_loop) == 1

    err = capsys.readouterr().err
    assert "======== App stopped by an error " in err
    assert "Unexpected error in test_loop_report.py, line" in err
    assert "  FileNotFoundError: Image file not found: assets/photo.jpg" in err
    assert "Traceback (most recent call last):" in err
    assert err.index("Unexpected error") < err.index("Traceback")


def test_app_error_in_user_loop_is_reported_with_hint(app, capsys):
    def user_loop():
        raise AppError("No speaker found", hint="Connect a speaker and restart the app.")

    assert app.loop(user_loop) == 1

    err = capsys.readouterr().err
    assert "Error in test_loop_report.py, line" in err
    assert "  No speaker found" in err
    assert "Hint: Connect a speaker and restart the app." in err


def test_stop_iteration_stops_the_loop_without_a_report(app, capsys):
    calls = []

    def user_loop():
        calls.append(None)
        if len(calls) == 3:
            raise StopIteration

    # The same exit code as a stop request: the launcher reads any code <= 128 as a failure
    assert app.loop(user_loop) == 128 + signal.SIGTERM
    assert len(calls) == 3

    captured = capsys.readouterr()
    assert "App stopped" not in captured.err
    assert "Traceback" not in captured.err + captured.out
