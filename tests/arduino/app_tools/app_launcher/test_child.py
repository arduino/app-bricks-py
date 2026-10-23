# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

import os

import pytest

from arduino.app_tools.app_launcher import child
from arduino.app_tools.app_launcher.child import Launch, Runner, environment, reseed, runner


def launch(app, env, **fields):
    return Launch(app=str(app), env=env, **fields)


def test_streamlit_apps_are_served_through_streamlit(tmp_path):
    assert runner(str(tmp_path)) is Runner.PYTHON
    (tmp_path / "app.yaml").write_text("bricks:\n  - arduino:streamlit_ui: {}\n")
    assert runner(str(tmp_path)) is Runner.STREAMLIT


def test_environment_is_the_launch_one_unbuffered(tmp_path):
    env = environment(launch(tmp_path, {"HOME": "/home/app", "PYTHONUNBUFFERED": "0"}))
    assert env == {"HOME": "/home/app", "PYTHONUNBUFFERED": "1"}


def test_environment_adds_the_app_bricks_to_pythonpath_once(tmp_path):
    bricks = str(tmp_path / "bricks")
    (tmp_path / "bricks").mkdir()
    env = environment(launch(tmp_path, {"PYTHONPATH": "/lib"}))
    assert env["PYTHONPATH"] == os.pathsep.join(["/lib", bricks])
    # run.sh exported it already when the folder existed at the container start
    assert environment(launch(tmp_path, {"PYTHONPATH": env["PYTHONPATH"]}))["PYTHONPATH"] == env["PYTHONPATH"]


def test_environment_activates_the_app_venv_the_launcher_runs_in(tmp_path, monkeypatch):
    venv = str(tmp_path / ".cache" / ".venv")
    monkeypatch.setattr(child, "in_venv", lambda path: path == venv)
    env = environment(launch(tmp_path, {"PATH": "/usr/bin"}))
    assert env["VIRTUAL_ENV"] == venv
    assert env["PATH"] == os.pathsep.join([os.path.join(venv, "bin"), "/usr/bin"])
    assert "VIRTUAL_ENV" not in environment(launch(tmp_path / "other", {"PATH": "/usr/bin"}))


def test_the_child_reseeds_numpy_global_generator():
    numpy = pytest.importorskip("numpy")
    numpy.random.seed(0)
    inherited = numpy.random.random()
    numpy.random.seed(0)
    reseed()
    assert numpy.random.random() != inherited
