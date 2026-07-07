# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""Cloud variable objects and sync policies for the Arduino Cloud brick.

This module reimplements the public surface that used to come from the
``arduino_iot_cloud`` library (``Location``, ``Color``, ``ColoredLight``,
``DimmedLight``, ``Schedule``) without depending on it. The brick no longer
speaks MQTT itself: connectivity, provisioning and the cloud handshake are
owned by the ``arduino-cloud-connector`` daemon, and the brick exchanges variable
values with it over a localhost REST + SSE API. These objects are therefore
pure value holders plus the per-variable conflict-resolution logic.

Conflict resolution is done here, in the brick, because the daemon is
deliberately policy-agnostic: it only delivers a value together with the
timestamp at which it was last set (locally or by the cloud). Each variable
carries a sync policy mirroring the C++ ``ArduinoIoTCloud`` semantics:

* ``CLOUD_WINS`` (default): an incoming cloud value is always applied when it
  differs from the local one.
* ``MOST_RECENT_WINS``: a cloud value is applied only if its timestamp is newer
  than the timestamp of the last local change.
* ``DEVICE_WINS``: cloud values are never applied; the local value is pushed
  back to the cloud so it converges to the device.
"""

import time
from typing import Any

# ── Sync policies ───────────────────────────────────────────────────────────
# String constants (JSON/log friendly) mirroring the C++ ArduinoIoTCloud enum.
DEVICE_WINS = "DEVICE_WINS"
CLOUD_WINS = "CLOUD_WINS"
MOST_RECENT_WINS = "MOST_RECENT_WINS"

_POLICIES = (DEVICE_WINS, CLOUD_WINS, MOST_RECENT_WINS)


def _now() -> float:
    """Current wall-clock time as epoch seconds (UTC), used for local changes.

    The brick runs on the same host as the daemon, so this is directly
    comparable to the timestamps the daemon stamps on values.
    """
    return time.time()


class CloudObject:
    """A single cloud variable, scalar or complex.

    A *scalar* object holds one value (bool/int/float/str). A *complex* object
    (created via the ``keys`` argument) holds a dict of scalar sub-objects, each
    exchanged with the daemon as an independent variable named ``"<name>:<key>"``
    (e.g. ``"clight:hue"``) — the same naming the cloud uses for structured
    properties.

    Callbacks (same contract as the legacy library):

    * ``on_write(client, value)`` — fired (in the brick loop) after a cloud
      update has been applied to this variable.
    * ``on_read(client) -> value`` — polled in the brick loop; its return value
      becomes the local value and is pushed to the cloud when it changes.
    * ``on_run(client, args)`` — polled in the brick loop unconditionally.
    """

    def __init__(self, name: str, **kwargs: Any):
        self.name = name
        self.on_read = kwargs.pop("on_read", None)
        self.on_write = kwargs.pop("on_write", None)
        self.on_run = kwargs.pop("on_run", None)
        self.interval = kwargs.pop("interval", 1.0)
        self.backoff = kwargs.pop("backoff", None)
        self.args = kwargs.pop("args", None)

        sync = kwargs.pop("sync", CLOUD_WINS)
        if sync not in _POLICIES:
            raise ValueError(f"invalid sync policy {sync!r}, expected one of {_POLICIES}")
        self.sync = sync

        value = kwargs.pop("value", None)
        keys = kwargs.pop("keys", None)

        # Internal state.
        self._owner = self  # whose on_write fires when this leaf changes
        self._push = None  # set by CloudObject.bind: callable(name, value)
        self._local_ts = None  # epoch secs of the last local change
        self._cloud_ts = None  # epoch secs of the last applied cloud value
        self._pending = False  # True while no thing is assigned (thing_unavailable)
        self.on_write_scheduled = False
        self.last_poll = 0.0

        if keys:
            # Complex object: build a scalar sub-object per key. Sub-object
            # callbacks live on this parent, so a cloud change on any leaf
            # schedules this object's on_write with the whole object.
            self._value = {}
            for key in keys:
                sub = CloudObject(f"{name}:{key}", value=kwargs.pop(key, None), sync=self.sync)
                sub._owner = self
                self._value[key] = sub
        else:
            self._value = value

        if kwargs:  # any leftover kwarg is a typo / unsupported option
            raise TypeError(f"'{type(self).__name__}' got unexpected keyword argument(s): {list(kwargs)}")

        self.runnable = any((self.on_run, self.on_read, self.on_write))

    def __repr__(self) -> str:
        return f"{self._value}"

    # ── value access ─────────────────────────────────────────────────────────
    @property
    def is_complex(self) -> bool:
        return isinstance(self._value, dict)

    @property
    def value(self) -> Any:
        return self._value

    def __contains__(self, key: str) -> bool:
        return self.is_complex and key in self._value

    def __getattr__(self, attr: str) -> Any:
        # Reached only for names not found normally — sub-record access on a
        # complex object (e.g. clight.hue).
        value = self.__dict__.get("_value", None)
        if isinstance(value, dict) and attr in value:
            return value[attr].value
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{attr}'")

    def __setattr__(self, attr: str, value: Any):
        existing = self.__dict__.get("_value", None)
        if isinstance(existing, dict) and attr in existing:
            existing[attr].set_local(value)  # clight.hue = 5 → push "clight:hue"
        else:
            super().__setattr__(attr, value)

    # ── transport binding ─────────────────────────────────────────────────────
    def bind(self, push):
        """Wire the daemon push callback into every scalar leaf of this object."""
        if self.is_complex:
            for sub in self._value.values():
                sub.bind(push)
        else:
            self._push = push

    def leaves(self) -> list["CloudObject"]:
        """Return the scalar leaves (the actual cloud variables) of this object."""
        if self.is_complex:
            return list(self._value.values())
        return [self]

    # ── local (device → cloud) changes ─────────────────────────────────────────
    def set_local(self, value: Any):
        """Apply a value set by the application and push it to the cloud."""
        if self.is_complex:
            # Assigning the whole object: accept a dict or another CloudObject.
            src = value._value if isinstance(value, CloudObject) else value
            if isinstance(src, dict):
                for key, sub in self._value.items():
                    if key in src:
                        sub.set_local(src[key])
            return

        value = self._coerce(value)
        if value is None or value == self._value:
            return
        self._value = value
        self._local_ts = _now()
        if self._pending:
            # No thing assigned yet: the daemon would reject the push
            # (thing_unavailable). Keep the value locally; it is pushed at sync
            # time according to the policy (lastvalue_missing → always;
            # DEVICE_WINS / MOST_RECENT_WINS when the local value should win).
            return
        if self._push is not None:
            self._push(self.name, value)

    def _coerce(self, value: Any) -> Any:
        # Workaround for the cloud int/float ambiguity: keep a float variable a
        # float even when assigned an int.
        if isinstance(self._value, float) and isinstance(value, int) and not isinstance(value, bool):
            return float(value)
        return value

    # ── cloud (cloud → device) changes ──────────────────────────────────────────
    def apply_cloud(self, value: Any, cloud_ts: float) -> bool:
        """Apply an incoming cloud value according to the sync policy.

        Returns True if the local value changed (so the caller schedules
        on_write). ``cloud_ts`` is epoch seconds (the daemon's last-value
        timestamp for this variable).
        """
        if cloud_ts is None:
            cloud_ts = _now()
        self._cloud_ts = cloud_ts
        value = self._coerce(value)

        if self.sync == DEVICE_WINS:
            # The device value always wins: ignore the cloud value and push the
            # local one back so the cloud converges. Re-push only on a real
            # divergence to avoid an echo loop.
            if self._value is not None and value != self._value and self._push is not None:
                self._push(self.name, self._value)
            return False

        if self.sync == MOST_RECENT_WINS and self._local_ts is not None and cloud_ts <= self._local_ts:
            # The local change is newer: keep it and push it up so the cloud
            # converges (guarded on divergence to avoid an echo loop).
            if self._value is not None and value != self._value and self._push is not None:
                self._push(self.name, self._value)
            return False

        if value == self._value:
            return False
        self._value = value
        return True

    def apply_missing(self):
        """Resolve a ``lastvalue_missing`` sync frame: the cloud has no stored
        value for this variable, so the local value wins and is pushed up so the
        cloud converges. Applies to every sync policy. No-op if there is no local
        value to assert yet.
        """
        if self._value is not None and self._push is not None:
            self._push(self.name, self._value)

    # ── loop execution ─────────────────────────────────────────────────────────
    def run_sync(self, client):
        """Run this object's callbacks once (called from the brick loop)."""
        if self.on_run is not None:
            self.on_run(client, self.args)
        if self.on_read is not None:
            self.set_local(self.on_read(client))
        if self.on_write is not None and self.on_write_scheduled:
            self.on_write_scheduled = False
            self.on_write(client, self if self.is_complex else self._value)


# ── Re-exported complex objects (formerly from arduino_iot_cloud) ────────────


class Location(CloudObject):
    def __init__(self, name: str, **kwargs: Any):
        super().__init__(name, keys=("lat", "lon"), **kwargs)


class Color(CloudObject):
    def __init__(self, name: str, **kwargs: Any):
        super().__init__(name, keys=("hue", "sat", "bri"), **kwargs)


class ColoredLight(CloudObject):
    def __init__(self, name: str, **kwargs: Any):
        super().__init__(name, keys=("swi", "hue", "sat", "bri"), **kwargs)


class DimmedLight(CloudObject):
    def __init__(self, name: str, **kwargs: Any):
        super().__init__(name, keys=("swi", "bri"), **kwargs)


class Schedule(CloudObject):
    """A cloud schedule (frm/to/len/msk). Computes its active state in on_run."""

    def __init__(self, name: str, **kwargs: Any):
        self.on_active = kwargs.pop("on_active", None)
        self.active = False
        kwargs["on_run"] = self._on_run
        super().__init__(name, keys=("frm", "to", "len", "msk"), **kwargs)

    def _initialized(self) -> bool:
        return all(sub.value is not None for sub in self._value.values())

    def _on_run(self, client, args=None):
        if not self._initialized():
            return
        ts = int(_now()) + (client.get("tz_offset", 0) if client is not None else 0)
        frm = self._value["frm"].value
        length = self._value["len"].value
        if frm < ts < (frm + length):
            if not self.active and self.on_active is not None:
                self.on_active(client, self)
            self.active = True
        else:
            self.active = False
