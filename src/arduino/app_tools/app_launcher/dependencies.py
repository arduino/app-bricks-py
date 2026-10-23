# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""The dependencies run.sh installs for an app and their fingerprint: when it changes, the container must be prepared again."""

import glob
import hashlib
import os
from contextlib import suppress

STREAMLIT_UI = "arduino:streamlit_ui"


def uses_streamlit(app: str) -> bool:
    """Tells whether the app.yaml uses the streamlit_ui brick, as run.sh checks."""
    try:
        with open(os.path.join(app, "app.yaml"), encoding="utf-8") as f:
            return STREAMLIT_UI in f.read()
    except OSError:
        return False


def dependencies(app: str) -> list[tuple[str, bytes]]:
    """Returns what the app needs beyond the image as (name, content) pairs, empty when the image is enough."""
    found: list[tuple[str, bytes]] = []
    pattern = os.path.join(glob.escape(app), "bricks", "*", "requirements.txt")
    for path in [os.path.join(app, "python", "requirements.txt"), *sorted(glob.glob(pattern))]:
        try:
            with open(path, "rb") as f:
                content = f.read()
        except OSError:
            continue
        if content.strip():
            found.append((os.path.relpath(path, app), content))
    # run.sh renames each installed wheel to x.whl.installed
    with suppress(OSError):
        wheels = {name.removesuffix(".installed") for name in os.listdir(os.path.join(app, "python-libraries"))}
        found.extend((f"python-libraries/{name}", b"") for name in sorted(wheels) if name.endswith(".whl"))
    if uses_streamlit(app):
        found.append(("app.yaml", STREAMLIT_UI.encode()))
    return found


def fingerprint(app: str) -> str | None:
    """Returns the digest of the app dependencies, None when it has none of its own."""
    found = dependencies(app)
    if not found:
        return None
    digest = hashlib.sha256()
    for name, content in found:
        digest.update(f"{name}\0{len(content)}\0".encode())
        digest.update(content)
    return digest.hexdigest()


def venv_dir(app: str) -> str:
    """Returns the app venv folder, which run.sh creates."""
    return os.path.join(app, ".cache", ".venv")
