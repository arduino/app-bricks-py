# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""Report the bricks of an app whose module name is also the name of something else the app can import.

A custom brick, in the bricks folder of the app, and a third-party brick, installed from the bricks registry,
are imported by the name of their folder: `from <brick_name> import ...`. The name may also be the one of a
module of the app, of another brick, of a standard library module or of an installed package: Python then loads
the first it finds and says nothing. This script asks the import system which one that is, for every brick, and
prints a warning for each brick that is hidden or hides something else. It reports and never fails: the app
starts in any case.

run.sh runs it right before the app, once the requirements are installed, and only for an app that has a custom
or a third-party brick:

    python -I check_brick_conflicts.py <app python dir> <custom bricks dir> <third-party bricks dir>

In isolated mode (-I) neither PYTHONPATH nor the folder of this script reach sys.path, so that a brick named
like a module this script imports cannot break it. The search path of the app is rebuilt here: the folder of
its main script, PYTHONPATH as run.sh exported it, then sys.path of the interpreter.

Only the standard library is imported, and as little of it as possible: this runs at every start of the app.
"""

import os
import sys
from importlib.machinery import BuiltinImporter, FrozenImporter, PathFinder


def _subfolders(folder: str) -> list[str]:
    try:
        return sorted(entry.path for entry in os.scandir(folder) if entry.is_dir())
    except OSError:
        return []


def _third_party_brick_dirs(third_party_dir: str) -> list[str]:
    """The folders of the mounted bricks, <third-party dir>/<mount>/<brick_name>, in the order run.sh lists them."""
    return [brick_dir for mount in _subfolders(third_party_dir) for brick_dir in _subfolders(mount)]


def _brick_id(brick_dir: str) -> str:
    """The id of the brick, `<namespace>:<brick_name>`, read from brick_config.yaml; the folder name without it."""
    try:
        with open(os.path.join(brick_dir, "brick_config.yaml"), encoding="utf-8") as config:
            for line in config:
                if line.startswith("id:"):
                    value = line[3:].split("#", 1)[0].strip().strip("\"'")
                    if value:
                        return value
    except OSError:
        pass
    return os.path.basename(brick_dir)


def _is_within(path: str, folder: str) -> bool:
    return path == folder or path.startswith(folder.rstrip(os.sep) + os.sep)


def find_conflicts(
    app_python_dir: str, custom_bricks_dir: str, third_party_dir: str, python_path: list[str], interpreter_path: list[str]
) -> list[str]:
    """One message for each brick that Python does not load by its name, or that hides something else."""
    search_path = [app_python_dir, *python_path, *interpreter_path]
    bricks = [(brick_dir, f"The custom brick '{os.path.basename(brick_dir)}'") for brick_dir in _subfolders(custom_bricks_dir)]
    bricks += [(brick_dir, f"The brick '{_brick_id(brick_dir)}'") for brick_dir in _third_party_brick_dirs(third_party_dir)]
    messages = []
    for brick_dir, brick in bricks:
        name = os.path.basename(brick_dir)
        if not os.path.isfile(os.path.join(brick_dir, "__init__.py")):
            continue  # Not a package, so not a brick: a folder of data, __pycache__

        # Modules built into the interpreter are found before any folder is searched
        if BuiltinImporter.find_spec(name) is not None or FrozenImporter.find_spec(name) is not None:
            messages.append(f"{brick} cannot be imported: '{name}' is a module built into Python, which is loaded in its place.")
            continue

        loaded = PathFinder.find_spec(name, search_path)
        if loaded is None or loaded.origin is None:
            continue  # A folder with a name Python cannot import
        if not _is_within(loaded.origin, brick_dir):
            messages.append(f"{brick} is hidden: `import {name}` loads {loaded.origin}, which comes first.")
            continue

        if name in sys.stdlib_module_names:
            messages.append(f"{brick} replaces the standard library module '{name}' for the whole app.")
            continue

        # The same name without the folder the brick is in: what the brick takes the place of.
        # A third-party brick is left out, it gets its own "is hidden" message.
        mount_dir = os.path.dirname(brick_dir)
        hidden = PathFinder.find_spec(name, [entry for entry in search_path if entry != mount_dir])
        if hidden is None:
            continue
        if hidden.origin is not None:
            what = "" if _is_within(hidden.origin, third_party_dir) else hidden.origin
        else:
            # A package without __init__.py, like `arduino`: it counts when it is an installed one,
            # not when it is a data folder of the app that happens to have the name of the brick
            installed = [
                folder for folder in hidden.submodule_search_locations or [] if any(_is_within(folder, entry) for entry in interpreter_path if entry)
            ]
            what = installed[0] if installed else ""
        if what:
            messages.append(f"{brick} hides {what}: `import {name}` loads the brick.")
    return messages


def main(argv: list[str]) -> int:
    if len(argv) != 4:
        print(f"usage: {argv[0]} <app python dir> <custom bricks dir> <third-party bricks dir>")
        return 2
    python_path = [entry for entry in os.environ.get("PYTHONPATH", "").split(os.pathsep) if entry]
    for message in find_conflicts(argv[1], argv[2], argv[3], python_path, list(sys.path)):
        print(f"Warning: {message}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
