# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""check_brick_conflicts.py of python-apps-base warns about the third-party bricks Python does not load by their name.

The script runs the way run.sh runs it, in isolated mode with PYTHONPATH as run.sh builds it: the folder of the
custom bricks first, then one folder for each mounted brick.
"""

import os
import subprocess
import sys
from pathlib import Path

CHECK = Path(__file__).parents[2] / "containers" / "bricks" / "python-apps-base" / "scripts" / "check_brick_conflicts.py"


def _brick(tmp_path: Path, mount: str, name: str, brick_id: str | None = None) -> Path:
    """A brick as arduino-app-cli mounts it: <third-party dir>/<random_string>/<brick_name>/."""
    brick = tmp_path / "third-party-bricks" / mount / name
    brick.mkdir(parents=True)
    (brick / "__init__.py").write_text("")
    if brick_id is not None:
        (brick / "brick_config.yaml").write_text(f"# a comment\nid: {brick_id}\nname: A brick\n")
    return brick


def _app_file(tmp_path: Path, relative: str) -> Path:
    path = tmp_path / "app" / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("")
    return path


def _check(tmp_path: Path) -> list[str]:
    """Run the check on the app and the bricks under tmp_path; return the lines it printed."""
    app = tmp_path / "app"
    (app / "python").mkdir(parents=True, exist_ok=True)
    third_party = tmp_path / "third-party-bricks"
    python_path = [str(app / "bricks")] if (app / "bricks").is_dir() else []
    if third_party.is_dir():
        python_path += sorted(str(mount) for mount in third_party.iterdir())
    result = subprocess.run(
        [sys.executable, "-I", str(CHECK), str(app / "python"), str(third_party)],
        env={**os.environ, "PYTHONPATH": os.pathsep.join(python_path)},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stderr == ""
    return result.stdout.splitlines()


def test_bricks_with_names_of_their_own_are_not_reported(tmp_path: Path):
    _brick(tmp_path, "a1b2c3", "soil_sensor", "acme:soil_sensor")
    _brick(tmp_path, "d4e5f6", "camera_tools", "zeta:camera_tools")
    _app_file(tmp_path, "python/helpers.py")
    _app_file(tmp_path, "bricks/mine/__init__.py")
    assert _check(tmp_path) == []


def test_no_third_party_bricks_no_output(tmp_path: Path):
    assert _check(tmp_path) == []


def test_a_module_of_the_app_hides_the_brick(tmp_path: Path):
    _brick(tmp_path, "a1b2c3", "soil_sensor", "acme:soil_sensor")
    module = _app_file(tmp_path, "python/soil_sensor.py")
    assert _check(tmp_path) == [f"Warning: The brick 'acme:soil_sensor' is hidden: `import soil_sensor` loads {module}, which comes first."]


def test_a_package_of_the_app_hides_the_brick(tmp_path: Path):
    _brick(tmp_path, "a1b2c3", "soil_sensor", "acme:soil_sensor")
    package = _app_file(tmp_path, "python/soil_sensor/__init__.py")
    assert _check(tmp_path) == [f"Warning: The brick 'acme:soil_sensor' is hidden: `import soil_sensor` loads {package}, which comes first."]


def test_a_data_folder_of_the_app_does_not_hide_the_brick(tmp_path: Path):
    # A folder without __init__.py is not a package: Python goes on and finds the brick
    _brick(tmp_path, "a1b2c3", "soil_sensor", "acme:soil_sensor")
    _app_file(tmp_path, "python/soil_sensor/readings.csv")
    assert _check(tmp_path) == []


def test_a_custom_brick_hides_the_brick(tmp_path: Path):
    _brick(tmp_path, "a1b2c3", "soil_sensor", "acme:soil_sensor")
    custom = _app_file(tmp_path, "bricks/soil_sensor/__init__.py")
    assert _check(tmp_path) == [f"Warning: The brick 'acme:soil_sensor' is hidden: `import soil_sensor` loads {custom}, which comes first."]


def test_of_two_bricks_with_the_same_name_the_second_is_reported_once(tmp_path: Path):
    first = _brick(tmp_path, "a1b2c3", "soil_sensor", "acme:soil_sensor")
    _brick(tmp_path, "d4e5f6", "soil_sensor", "zeta:soil_sensor")
    assert _check(tmp_path) == [
        f"Warning: The brick 'zeta:soil_sensor' is hidden: `import soil_sensor` loads {first / '__init__.py'}, which comes first."
    ]


def test_a_brick_named_like_a_module_built_into_python_cannot_be_imported(tmp_path: Path):
    _brick(tmp_path, "a1b2c3", "time", "acme:time")  # compiled into the interpreter
    _brick(tmp_path, "d4e5f6", "os", "acme:os")  # frozen into the interpreter
    assert _check(tmp_path) == [
        "Warning: The brick 'acme:time' cannot be imported: 'time' is a module built into Python, which is loaded in its place.",
        "Warning: The brick 'acme:os' cannot be imported: 'os' is a module built into Python, which is loaded in its place.",
    ]


def test_a_brick_named_like_a_standard_library_module_replaces_it(tmp_path: Path):
    _brick(tmp_path, "a1b2c3", "json", "acme:json")
    assert _check(tmp_path) == ["Warning: The brick 'acme:json' replaces the standard library module 'json' for the whole app."]


def test_a_brick_named_like_an_installed_package_hides_it(tmp_path: Path):
    import pytest  # installed in the environment the check runs in

    _brick(tmp_path, "a1b2c3", "pytest", "acme:pytest")
    assert _check(tmp_path) == [f"Warning: The brick 'acme:pytest' hides {pytest.__file__}: `import pytest` loads the brick."]


def test_a_brick_named_like_an_installed_package_without_init_hides_it(tmp_path: Path):
    # `arduino`, the package of the framework, has no __init__.py: a brick with that name takes its place
    site = tmp_path / "site-packages"
    (site / "arduino" / "app_utils").mkdir(parents=True)
    (site / "arduino" / "app_utils" / "__init__.py").write_text("")
    _brick(tmp_path, "a1b2c3", "arduino", "acme:arduino")
    _app_file(tmp_path, "python/placeholder.py")
    sys.path.insert(0, str(CHECK.parent))
    try:
        from check_brick_conflicts import find_conflicts
    finally:
        sys.path.pop(0)
    third_party = tmp_path / "third-party-bricks"
    messages = find_conflicts(str(tmp_path / "app" / "python"), str(third_party), [str(third_party / "a1b2c3")], [str(site)])
    assert messages == [f"The brick 'acme:arduino' hides {site / 'arduino'}: `import arduino` loads the brick."]


def test_a_brick_without_an_id_is_named_after_its_folder(tmp_path: Path):
    _brick(tmp_path, "a1b2c3", "json")
    assert _check(tmp_path) == ["Warning: The brick 'json' replaces the standard library module 'json' for the whole app."]


def test_a_missing_third_party_folder_is_not_an_error(tmp_path: Path):
    result = subprocess.run([sys.executable, "-I", str(CHECK), str(tmp_path / "python"), str(tmp_path / "nope")], capture_output=True, text=True)
    assert (result.returncode, result.stdout, result.stderr) == (0, "", "")
