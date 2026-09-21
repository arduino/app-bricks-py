# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""Available memory, as seen from inside the container.

It is the minimum of:
  - the system MemAvailable (/proc/meminfo);
  - the space left within the container limit (cgroup v2 or v1), if any.
Reclaimable file cache is not counted as used.
"""

from pathlib import Path

CGROUP_V2 = Path("/sys/fs/cgroup")
CGROUP_V1 = Path("/sys/fs/cgroup/memory")


def available_bytes() -> int | None:
    """Available bytes, or None if the system does not provide the information."""
    values = [v for v in (_system_available(), _cgroup_available()) if v is not None]
    return min(values) if values else None


def _system_available() -> int | None:
    kb = _read_field(Path("/proc/meminfo"), "MemAvailable:")
    return None if kb is None else kb * 1024


def _cgroup_available() -> int | None:
    limit = _read_int(CGROUP_V2 / "memory.max")
    if limit is not None:  # cgroup v2
        usage = _read_int(CGROUP_V2 / "memory.current")
        reclaimable = _read_field(CGROUP_V2 / "memory.stat", "inactive_file")
    else:  # cgroup v1
        limit = _read_int(CGROUP_V1 / "memory.limit_in_bytes")
        if limit is None or limit >= 1 << 60:  # huge value = no limit
            return None
        usage = _read_int(CGROUP_V1 / "memory.usage_in_bytes")
        reclaimable = _read_field(CGROUP_V1 / "memory.stat", "total_inactive_file")
    if usage is None:
        return None
    return max(0, limit - (usage - (reclaimable or 0)))


def _read_int(path: Path) -> int | None:
    try:
        text = path.read_text().strip()
        return None if text == "max" else int(text)
    except (OSError, ValueError):
        return None


def _read_field(path: Path, key: str) -> int | None:
    """First number on the line starting with key."""
    try:
        for line in path.read_text().splitlines():
            fields = line.split()
            if fields and fields[0] == key:
                return int(fields[1])
    except (OSError, ValueError, IndexError):
        pass
    return None
