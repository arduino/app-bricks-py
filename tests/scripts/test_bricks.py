# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""Unit tests for the bricks listing and scaffold."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

# Make the repo-root ``scripts`` package importable regardless of the cwd.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.bricks import BRICKS_DIR, TESTS_DIR, BricksError, describe, list_text, load_bricks, main, scaffold  # noqa: E402


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    bricks = tmp_path / BRICKS_DIR
    for name, config in {
        "wave_generator": {
            "id": "arduino:wave_generator",
            "name": "Wave Generator",
            "description": "Waves",
            "category": "audio",
            "required_devices": ["speaker"],
        },
        "llm": {"id": "arduino:llm", "name": "LLM", "description": "Chat", "category": "ai"},
    }.items():
        (bricks / name).mkdir(parents=True)
        (bricks / name / "brick_config.yaml").write_text(yaml.safe_dump(config))
    (bricks / "llm" / "brick_compose.yaml").write_text("services: {}\n")
    (bricks / "_shared").mkdir()
    (tmp_path / TESTS_DIR / "llm").mkdir(parents=True)
    return tmp_path


def test_list_reads_every_brick_config(repo: Path) -> None:
    bricks = load_bricks(repo / BRICKS_DIR)
    assert list(bricks) == ["llm", "wave_generator"], "sorted, directories without a config are not bricks"
    assert list_text(bricks).splitlines() == ["llm             ai     LLM", "wave_generator  audio  Wave Generator"]


def test_show_reports_config_files_and_tests(repo: Path) -> None:
    bricks = load_bricks(repo / BRICKS_DIR)
    text = describe(repo, bricks, "llm")
    assert "  id:          arduino:llm" in text
    assert "  containers:  brick_compose.yaml" in text
    assert f"  tests:       {TESTS_DIR / 'llm'}" in text
    text = describe(repo, bricks, "wave_generator")
    assert "  devices:     speaker" in text
    assert "  containers:  -" in text
    assert "  tests:       -" in text
    with pytest.raises(BricksError, match="Unknown brick"):
        describe(repo, bricks, "nope")


def test_new_brick_creates_package_config_readme_and_test(repo: Path) -> None:
    steps = scaffold(repo, "mood_light", None, "Drives a light from the mood", "ui")

    package = repo / BRICKS_DIR / "mood_light"
    assert (package / "__init__.py").read_text().endswith('from .mood_light import *\n\n__all__ = ["MoodLight"]\n')
    module = (package / "mood_light.py").read_text()
    assert module.startswith("# SPDX-FileCopyrightText")
    assert "@brick\nclass MoodLight:" in module
    assert '"""Drives a light from the mood"""' in module
    assert yaml.safe_load((package / "brick_config.yaml").read_text()) == {
        "id": "arduino:mood_light",
        "name": "Mood Light",
        "description": "Drives a light from the mood",
        "category": "ui",
    }
    assert (package / "README.md").read_text().startswith("# Mood Light brick\n\nDrives a light from the mood\n")
    assert "from arduino.app_bricks.mood_light import MoodLight" in (repo / TESTS_DIR / "mood_light" / "test_mood_light.py").read_text()
    assert load_bricks(repo / BRICKS_DIR)["mood_light"]["name"] == "Mood Light"
    assert any("task test:bricks" in step for step in steps)


def test_new_brick_honours_display_name(repo: Path) -> None:
    scaffold(repo, "tts2", "Text to Speech v2", "Speaks", "audio")
    assert yaml.safe_load((repo / BRICKS_DIR / "tts2" / "brick_config.yaml").read_text())["name"] == "Text to Speech v2"


@pytest.mark.parametrize(("name", "category"), [("Bad-Name", "ui"), ("1abc", "ui"), ("llm", "ai"), ("fine", "nope")])
def test_new_brick_rejects_invalid_names_and_categories(repo: Path, name: str, category: str) -> None:
    with pytest.raises(BricksError):
        scaffold(repo, name, None, "x", category)
    assert not (repo / TESTS_DIR / name).exists() or name == "llm"


def test_cli(repo: Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = ["--repo-root", str(repo)]
    assert main([*root, "list"]) == 0
    assert "wave_generator" in capsys.readouterr().out
    assert main([*root, "show", "llm"]) == 0
    assert "arduino:llm" in capsys.readouterr().out
    assert main([*root, "new", "fresh", "--desc", "New one", "--category", "text"]) == 0
    assert capsys.readouterr().out.startswith(f"Scaffolded {BRICKS_DIR / 'fresh'}. Next steps:")
    assert main([*root, "new", "fresh"]) == 1
    assert "already exists" in capsys.readouterr().err
    assert main([*root, "show", "nope"]) == 1
