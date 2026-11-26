# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""Every enabled brick declares a category from the shared list."""

from pathlib import Path

import pytest
import yaml

from arduino.app_tools.brick_categories import BRICK_CATEGORIES, DEFAULT_BRICK_CATEGORY

BRICKS_DIR = Path(__file__).resolve().parents[3] / "src" / "arduino" / "app_bricks"
BRICK_CONFIGS = sorted(BRICKS_DIR.glob("*/brick_config.yaml"))


def test_category_list_is_sorted_and_unique() -> None:
    assert list(BRICK_CATEGORIES) == sorted(set(BRICK_CATEGORIES))
    assert DEFAULT_BRICK_CATEGORY in BRICK_CATEGORIES


def test_bricks_are_found() -> None:
    assert BRICK_CONFIGS


@pytest.mark.parametrize("config_file", BRICK_CONFIGS, ids=lambda path: path.parent.name)
def test_brick_declares_a_known_category(config_file: Path) -> None:
    config = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    if config.get("disabled"):
        pytest.skip("disabled bricks are hidden from App Lab")
    assert "category" in config, f"{config_file.parent.name} does not declare a category"
    assert config["category"] in BRICK_CATEGORIES, (
        f"{config_file.parent.name} declares '{config['category']}', use one of {', '.join(BRICK_CATEGORIES)}"
    )
