# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""The library bricks an app declares, as the modules the launcher preloads for it."""

import importlib.util
import os
import pathlib

import yaml


def _brick_id(config: pathlib.Path) -> str | None:
    """The id a brick_config.yaml declares, None when it declares none or cannot be read."""
    try:
        brick_id = (yaml.safe_load(config.read_text(encoding="utf-8")) or {}).get("id")
    except (OSError, yaml.YAMLError, AttributeError):
        return None
    return brick_id if isinstance(brick_id, str) else None


def library_brick_module(brick_id: str) -> str | None:
    """The module of the library brick with the given id, found next to its brick_config.yaml; None for a brick the library does not ship.

    The package is usually named after the id, so that folder is read first; only a brick whose package is named otherwise costs a
    read of every brick_config.yaml.
    """
    vendor, _, name = brick_id.partition(":")
    spec = importlib.util.find_spec("arduino.app_bricks") if vendor == "arduino" and name else None
    locations = [pathlib.Path(location) for location in (spec.submodule_search_locations or [])] if spec else []
    candidates = [location / name / "brick_config.yaml" for location in locations]
    candidates += [config for location in locations for config in sorted(location.glob("*/brick_config.yaml"))]
    for config in candidates:
        if config.is_file() and _brick_id(config) == brick_id:
            return f"arduino.app_bricks.{config.parent.name}"
    return None


def app_bricks_modules(app_dir: str) -> list[str]:
    """The modules of the library bricks an app declares in its app.yaml, in declaration order; nothing for a missing or unreadable file."""
    try:
        with open(os.path.join(app_dir, "app.yaml"), encoding="utf-8") as f:
            descriptor = yaml.safe_load(f) or {}
    except (OSError, yaml.YAMLError):
        return []
    modules: list[str] = []
    for entry in descriptor.get("bricks") or [] if isinstance(descriptor, dict) else []:
        for brick_id in [entry] if isinstance(entry, str) else list(entry) if isinstance(entry, dict) else []:
            module = library_brick_module(str(brick_id))
            if module and module not in modules:
                modules.append(module)
    return modules
