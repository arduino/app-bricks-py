# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

import pytest

from arduino.app_tools.app_launcher import dependencies


def write(path, content=""):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def test_an_app_without_dependencies_has_no_fingerprint(tmp_path):
    write(tmp_path / "python" / "requirements.txt", "\n  \n")
    write(tmp_path / "bricks" / "mine" / "requirements.txt")
    write(tmp_path / "python-libraries" / "README.md", "no wheels")
    write(tmp_path / "app.yaml", "bricks:\n  - arduino:web_ui: {}\n")
    assert dependencies.fingerprint(str(tmp_path)) is None


@pytest.mark.parametrize(
    "dependency",
    [
        ("python/requirements.txt", "requests\n"),
        ("bricks/mine/requirements.txt", "numpy\n"),
        ("python-libraries/lib-1.0-py3-none-any.whl", ""),
        ("python-libraries/lib-1.0-py3-none-any.whl.installed", ""),
        ("app.yaml", "bricks:\n  - arduino:streamlit_ui: {}\n"),
    ],
)
def test_each_kind_of_dependency_gets_a_fingerprint(tmp_path, dependency):
    path, content = dependency
    write(tmp_path / path, content)
    assert dependencies.fingerprint(str(tmp_path)) is not None


def test_the_fingerprint_follows_the_requirements(tmp_path):
    requirements = tmp_path / "python" / "requirements.txt"
    write(requirements, "requests\n")
    first = dependencies.fingerprint(str(tmp_path))
    assert dependencies.fingerprint(str(tmp_path)) == first
    write(requirements, "requests==2.32.0\n")
    second = dependencies.fingerprint(str(tmp_path))
    assert second != first
    write(tmp_path / "bricks" / "mine" / "requirements.txt", "numpy\n")
    assert dependencies.fingerprint(str(tmp_path)) not in (first, second)


def test_installing_a_wheel_keeps_the_fingerprint(tmp_path):
    wheel = tmp_path / "python-libraries" / "lib-1.0-py3-none-any.whl"
    write(wheel)
    before = dependencies.fingerprint(str(tmp_path))
    wheel.rename(f"{wheel}.installed")
    assert dependencies.fingerprint(str(tmp_path)) == before
    write(tmp_path / "python-libraries" / "lib-1.1-py3-none-any.whl")
    assert dependencies.fingerprint(str(tmp_path)) != before


def test_streamlit_is_read_from_the_app_yaml(tmp_path):
    assert not dependencies.uses_streamlit(str(tmp_path))
    write(tmp_path / "app.yaml", "bricks:\n  - arduino:streamlit_ui: {}\n")
    assert dependencies.uses_streamlit(str(tmp_path))
