# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""The bricks of the library: list them, describe one, scaffold a new one.

  python3 -m scripts.bricks list
  python3 -m scripts.bricks show wave_generator
  python3 -m scripts.bricks new my_brick --name "My Brick" --desc "What it does" --category audio

A brick is a package under src/arduino/app_bricks/<name>/ with a brick_config.yaml, a README.md
and a module exposing its @brick classes, plus tests under tests/arduino/app_bricks/<name>/.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
BRICKS_DIR = Path("src/arduino/app_bricks")
TESTS_DIR = Path("tests/arduino/app_bricks")
CONFIG_FILE = "brick_config.yaml"

NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")
CATEGORIES = ("ai", "audio", "image", "miscellaneous", "storage", "text", "ui", "video")

SPDX_HEADER = """# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0
"""


class BricksError(RuntimeError):
    """Raised when the bricks cannot be read or a brick cannot be scaffolded."""


def load_bricks(bricks_dir: Path) -> dict[str, dict]:
    """Map every brick directory name to its brick_config.yaml content."""
    bricks: dict[str, dict] = {}
    for config_file in sorted(bricks_dir.glob(f"*/{CONFIG_FILE}")):
        try:
            config = yaml.safe_load(config_file.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError as error:
            raise BricksError(f"{config_file} is not valid YAML: {error}") from error
        bricks[config_file.parent.name] = config
    if not bricks:
        raise BricksError(f"No bricks found under {bricks_dir}.")
    return bricks


def list_text(bricks: dict[str, dict]) -> str:
    """One line per brick: directory name, category and display name."""
    width = max(len(name) for name in bricks)
    category_width = max(len(str(config.get("category", ""))) for config in bricks.values())
    return "\n".join(
        f"{name:<{width}}  {str(config.get('category', '')):<{category_width}}  {config.get('name', '')}" for name, config in bricks.items()
    )


def describe(repo_root: Path, bricks: dict[str, dict], name: str) -> str:
    """Render what the repository knows about one brick."""
    if name not in bricks:
        raise BricksError(f"Unknown brick '{name}', see `list`.")
    config = bricks[name]
    directory = repo_root / BRICKS_DIR / name
    compose_files = sorted(path.name for path in directory.glob("brick_compose*.yaml"))
    lines = [
        f"{name}",
        f"  id:          {config.get('id', '-')}",
        f"  name:        {config.get('name', '-')}",
        f"  category:    {config.get('category', '-')}",
        f"  description: {config.get('description', '-')}",
        f"  devices:     {', '.join(config.get('required_devices') or []) or '-'}",
        f"  directory:   {directory.relative_to(repo_root)}",
        f"  containers:  {', '.join(compose_files) or '-'}",
        f"  tests:       {TESTS_DIR / name if (repo_root / TESTS_DIR / name).is_dir() else '-'}",
    ]
    return "\n".join(lines)


def class_name(name: str) -> str:
    return "".join(part.capitalize() for part in name.split("_"))


def module_text(name: str, display_name: str, desc: str) -> str:
    cls = class_name(name)
    return f'''{SPDX_HEADER}
from arduino.app_utils import Logger, brick

logger = Logger("{cls}")


@brick
class {cls}:
    """{desc}"""

    def __init__(self) -> None:
        # TODO: constructor parameters are the brick's configuration, keep them few and typed
        pass

    def start(self) -> None:
        """Called by App.run() before the loop starts, or manually when created later."""
        logger.info("{display_name} started")

    def stop(self) -> None:
        """Called by App.run() on shutdown, or manually when created later."""
        logger.info("{display_name} stopped")
'''


def init_text(name: str) -> str:
    return f'{SPDX_HEADER}\nfrom .{name} import *\n\n__all__ = ["{class_name(name)}"]\n'


def config_text(name: str, display_name: str, desc: str, category: str) -> str:
    return yaml.safe_dump({"id": f"arduino:{name}", "name": display_name, "description": desc, "category": category}, sort_keys=False)


def readme_text(display_name: str, desc: str, name: str) -> str:
    return f"""# {display_name} brick

{desc}

## Overview

TODO: what the brick does and when to use it.

## Features

- TODO

## Usage

```python
from arduino.app_bricks.{name} import {class_name(name)}
from arduino.app_utils import App

{name} = {class_name(name)}()

App.run()
```
"""


def test_text(name: str) -> str:
    cls = class_name(name)
    return f"""{SPDX_HEADER}
from arduino.app_bricks.{name} import {cls}


def test_starts_and_stops() -> None:
    brick = {cls}()
    brick.start()
    brick.stop()
"""


def scaffold(repo_root: Path, name: str, display_name: str | None, desc: str, category: str) -> list[str]:
    """Create the brick package and its test directory, returning the next steps."""
    if not NAME_PATTERN.match(name):
        raise BricksError(f"'{name}' is not a valid brick name, use a lowercase Python identifier like my_brick.")
    if category not in CATEGORIES:
        raise BricksError(f"'{category}' is not a brick category, use one of {', '.join(CATEGORIES)}.")
    directory = repo_root / BRICKS_DIR / name
    tests = repo_root / TESTS_DIR / name
    if directory.exists() or tests.exists():
        raise BricksError(f"A brick named '{name}' already exists.")
    display_name = display_name or name.replace("_", " ").title()

    directory.mkdir(parents=True)
    (directory / "__init__.py").write_text(init_text(name), encoding="utf-8")
    (directory / f"{name}.py").write_text(module_text(name, display_name, desc), encoding="utf-8")
    (directory / CONFIG_FILE).write_text(config_text(name, display_name, desc, category), encoding="utf-8")
    (directory / "README.md").write_text(readme_text(display_name, desc, name), encoding="utf-8")
    tests.mkdir(parents=True)
    (tests / f"test_{name}.py").write_text(test_text(name), encoding="utf-8")

    return [
        f"Implement {BRICKS_DIR / name / (name + '.py')} and its tests in {TESTS_DIR / name}, then run `task test:bricks -- {TESTS_DIR / name}`.",
        f"Complete {BRICKS_DIR / name / 'README.md'} and the description in {CONFIG_FILE}.",
        "If the brick runs a container, add a brick_compose.yaml next to the config.",
        "If the brick needs extra packages, add an extra to pyproject.toml and run `task deps:lock`.",
    ]


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT, help=argparse.SUPPRESS)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("list", help="Print every brick with its category and display name.")
    show_parser = subparsers.add_parser("show", help="Print the configuration and files of one brick.")
    show_parser.add_argument("name")
    new_parser = subparsers.add_parser("new", help="Scaffold a brick package and its test directory.")
    new_parser.add_argument("name", help="Brick name, a lowercase Python identifier like my_brick.")
    new_parser.add_argument("--name", dest="display_name", help="Display name, defaults to the name in title case.")
    new_parser.add_argument("--desc", default="TODO", help="One line description for brick_config.yaml and the README.")
    new_parser.add_argument("--category", default="miscellaneous", choices=CATEGORIES, help="Category in brick_config.yaml.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = create_parser().parse_args(argv)
    try:
        if args.command == "new":
            steps = scaffold(args.repo_root, args.name, args.display_name, args.desc, args.category)
            print(f"Scaffolded {BRICKS_DIR / args.name}. Next steps:")
            for index, step in enumerate(steps, 1):
                print(f"  {index}. {step}")
            return 0
        bricks = load_bricks(args.repo_root / BRICKS_DIR)
        print(list_text(bricks) if args.command == "list" else describe(args.repo_root, bricks, args.name))
    except BricksError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
