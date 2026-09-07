# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""The pose vocabulary an app declares: one spec per active pose, parsed from names or dicts."""

from dataclasses import dataclass
from typing import Any, Literal, Self

_POSE_TYPES = ("state", "action")
_POSE_KEYS = ("name", "type", "duration", "thresholds", "smoothing")


def _as_number(value: object, what: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{what} must be a number, got {value!r}")
    return float(value)


@dataclass(frozen=True, kw_only=True)
class PoseSpec:
    """One pose of the vocabulary declared with the `poses` constructor argument.

    Attributes:
        name (str): The pose name.
        builtin (bool): Whether the pose is one of `BUILTIN_POSE_NAMES`.
        type (Literal["state", "action"]): "state" for a held pose, "action" for a
            movement with a start and an end. Built-in poses are always "state".
        duration (float | None): Typical length of one occurrence of an action, in seconds.
        enter (float | None): Vote share above which the pose is entered; None keeps the
            shipped value.
        exit (float | None): Vote share below which the pose is left; None keeps the
            shipped value.
        smoothing (float | None): Time constant of the moving average, in seconds; None
            keeps the default.
    """

    name: str
    builtin: bool
    type: Literal["state", "action"] = "state"
    duration: float | None = None
    enter: float | None = None
    exit: float | None = None
    smoothing: float | None = None

    @classmethod
    def from_item(cls, item: str | dict[str, Any], builtin_names: tuple[str, ...], custom_names: tuple[str, ...] = ()) -> Self:
        """Build a spec from one item of the `poses` list: a name, or a dict with `name` and options.

        custom_names are the pose folders found next to the built-in names; any other name is refused.
        """
        if isinstance(item, str):
            options: dict[str, Any] = {"name": item}
        elif isinstance(item, dict):
            options = dict(item)
        else:
            raise ValueError(f"each pose must be a name or a dict, got {item!r}")
        name = options.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError(f"pose {item!r}: 'name' must be a non-empty string")
        unknown = [key for key in options if key not in _POSE_KEYS]
        if unknown:
            raise ValueError(f"pose {name!r}: unknown key {unknown[0]!r} (allowed: {', '.join(_POSE_KEYS)})")
        builtin = name in builtin_names
        if not builtin and name not in custom_names:
            folders = f"; pose folders: {', '.join(custom_names)}" if custom_names else "; no pose folders found"
            raise ValueError(f"unknown pose {name!r} (built-in: {', '.join(builtin_names)}{folders})")
        pose_type = options.get("type", "state")
        if pose_type not in _POSE_TYPES:
            raise ValueError(f"pose {name!r}: unknown type {pose_type!r} (use one of {_POSE_TYPES})")
        if builtin and pose_type != "state":
            raise ValueError(f"pose {name!r}: built-in poses are held poses, their type cannot be changed")
        duration = None
        if "duration" in options:
            if pose_type != "action":
                raise ValueError(f"pose {name!r}: duration applies to actions only")
            duration = _as_number(options["duration"], f"pose {name!r}: duration")
            if not duration > 0.0:
                raise ValueError(f"pose {name!r}: duration must be positive seconds, got {options['duration']!r}")
        enter = exit_ = None
        if "thresholds" in options:
            thresholds = options["thresholds"]
            if not isinstance(thresholds, dict) or set(thresholds) != {"enter", "exit"}:
                raise ValueError(f"pose {name!r}: thresholds must be a dict with 'enter' and 'exit', got {thresholds!r}")
            enter = _as_number(thresholds["enter"], f"pose {name!r}: thresholds['enter']")
            exit_ = _as_number(thresholds["exit"], f"pose {name!r}: thresholds['exit']")
            if not 0.0 <= exit_ < enter <= 1.0:
                raise ValueError(f"pose {name!r}: thresholds must satisfy 0 <= exit < enter <= 1, got {thresholds!r}")
        smoothing = None
        if "smoothing" in options:
            smoothing = _as_number(options["smoothing"], f"pose {name!r}: smoothing")
            if not smoothing > 0.0:
                raise ValueError(f"pose {name!r}: smoothing must be positive seconds, got {options['smoothing']!r}")
        return cls(name=name, builtin=builtin, type=pose_type, duration=duration, enter=enter, exit=exit_, smoothing=smoothing)


def parse_poses(poses: list[str | dict[str, Any]] | None, builtin_names: tuple[str, ...], custom_names: tuple[str, ...] = ()) -> tuple[PoseSpec, ...]:
    """Normalize the `poses` constructor argument to one spec per active pose."""
    if poses is None:
        return tuple(PoseSpec(name=name, builtin=True) for name in builtin_names)
    if not isinstance(poses, (list, tuple)):
        raise ValueError(f"poses must be a list of pose names or dicts, got {poses!r}")
    if not poses:
        raise ValueError("poses must list at least one pose (None selects the built-in poses)")
    specs = tuple(PoseSpec.from_item(item, builtin_names, custom_names) for item in poses)
    names = [spec.name for spec in specs]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(f"poses declared more than once: {', '.join(duplicates)}")
    return specs
