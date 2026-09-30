# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""The one way model sizes are measured and reported: ``size_mb``.

Every event and listing entry that sizes a model — the listing, the ``info`` stat
event, the completion and "Model exists" events of every handler — reports it as
``size_mb``: MiB (1024 * 1024 bytes, the unit ``du -m`` and the models-list.yaml
``model_size_mb`` values use) rounded to two decimals. Sizes are summed in bytes and
rounded once, here, so the size a download reports on completion is exactly the one a
later listing gives the same files.

Bookkeeping files (the ".download" marker and the ".arduino_metadata.yaml" record) are
not model content and are never counted: a directory measures the same before and
after its record is written.

Run as a script, it prints the "Model exists" event of a handler whose downloader
decides that in the shell (AI Hub, Edge Impulse), sized like everything else::

    python /app/common/model_size.py --description "Model exists: foo" [--downloading false] <path>
"""

import argparse
import json
import os
import stat
import sys

# Also defined by common/download_marker.py: this module imports nothing from the
# package, so the shell scripts can run it as a plain script.
MARKER_NAME = ".download"
METADATA_NAME = ".arduino_metadata.yaml"

_MIB = 1024 * 1024


def is_bookkeeping_name(name):
    """True when *name* is a metadata/marker file rather than model content.

    Matches the ``.arduino_metadata.yaml.tmp`` sibling of an interrupted atomic
    write too, so a directory holding only that is still treated as incomplete.
    """
    return name == MARKER_NAME or name.startswith(METADATA_NAME)


def size_mb(size_bytes):
    """*size_bytes* in MiB rounded to two decimals, or None when the size is unknown."""
    if size_bytes is None:
        return None
    return round(size_bytes / _MIB, 2)


def path_size_bytes(path):
    """Bytes of model content at *path* (a file, or a directory tree), or None if missing.

    Symlinks are not followed and bookkeeping files are skipped; an unreadable
    subdirectory counts as empty rather than failing the whole measurement.
    """
    try:
        st = os.stat(path, follow_symlinks=False)
    except OSError:
        return None
    if stat.S_ISREG(st.st_mode):
        return st.st_size
    if not stat.S_ISDIR(st.st_mode):
        return None

    total = 0
    stack = [path]
    while stack:
        current = stack.pop()
        try:
            with os.scandir(current) as it:
                for entry in it:
                    try:
                        entry_stat = entry.stat(follow_symlinks=False)
                    except OSError:
                        continue
                    if stat.S_ISDIR(entry_stat.st_mode):
                        stack.append(entry.path)
                    elif stat.S_ISREG(entry_stat.st_mode) and not is_bookkeeping_name(entry.name):
                        total += entry_stat.st_size
        except OSError:
            continue
    return total


def paths_size_bytes(paths):
    """Total bytes of *paths*, or None when there are none or any of them is missing."""
    if not paths:
        return None
    total = 0
    for path in paths:
        size = path_size_bytes(path)
        if size is None:
            return None
        total += size
    return total


def paths_size_mb(paths):
    """``size_mb`` of *paths* taken together; see ``paths_size_bytes``."""
    return size_mb(paths_size_bytes(paths))


def exists_event(description, path, downloading=None):
    """The "Model exists" info event for the model at *path*."""
    data = {"event": "info", "description": description}
    if downloading is not None:
        data["downloading"] = downloading
    data["size_mb"] = size_mb(path_size_bytes(path))
    return data


def main(argv=None):
    parser = argparse.ArgumentParser(description="Print a 'Model exists' event sized with size_mb.")
    parser.add_argument("--description", required=True)
    parser.add_argument("--downloading", choices=("true", "false"), help="Also report the downloading flag.")
    parser.add_argument("path")
    args = parser.parse_args(argv)
    downloading = None if args.downloading is None else args.downloading == "true"
    print(json.dumps(exists_event(args.description, args.path, downloading)), flush=True)


if __name__ == "__main__":
    sys.exit(main())
