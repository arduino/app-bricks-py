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

from arduino.app_utils import Logger

logger = Logger("ArduinoCloud")

# ── Sync policies ───────────────────────────────────────────────────────────
# String constants (JSON/log friendly) mirroring the C++ ArduinoIoTCloud enum.
DEVICE_WINS = "DEVICE_WINS"
CLOUD_WINS = "CLOUD_WINS"
MOST_RECENT_WINS = "MOST_RECENT_WINS"

_POLICIES = (DEVICE_WINS, CLOUD_WINS, MOST_RECENT_WINS)

# ── Update policy (device → cloud) ───────────────────────────────────────────
# The ``interval`` argument mirrors the C++ ArduinoIoTCloud ``seconds`` argument
# of ``addProperty`` (see AIoTC_Const.h ``static long const ON_CHANGE = -1``):
#
# * ``interval == ON_CHANGE`` (a negative sentinel, the default): publish on
#   change — the value is sent to the cloud whenever it differs from the last
#   value sent, throttled to at most one publish per ``_MIN_PUBLISH_INTERVAL``
#   (mirrors ``publishOnChange`` with Property::DEFAULT_MIN_TIME_BETWEEN_UPDATES).
# * ``interval > 0``: publish on a timer — the current value is sent every
#   ``interval`` seconds regardless of whether it changed (mirrors
#   ``publishEvery(seconds)``). Values that change within a window are not
#   individually sent; only the latest value at the tick is published.
ON_CHANGE = -1.0

# Minimum time between two device→cloud publishes in ON_CHANGE mode, in seconds.
# Mirrors the C++ Property::DEFAULT_MIN_TIME_BETWEEN_UPDATES_MILLIS = 500 ms (2 Hz).
_MIN_PUBLISH_INTERVAL = 0.5


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

    * ``on_write(client, value)`` — fired immediately, as soon as a cloud update
      has been applied to this variable (from the SSE listener), mirroring the
      synchronous ``onUpdate`` dispatch of the C++ ArduinoIoTCloud library. It is
      NOT gated by ``interval``.
    * ``on_read(client) -> value`` — polled in the brick loop; its return value
      becomes the local value, published to the cloud per the update policy (see
      ``interval`` / ``ON_CHANGE`` above). In ON_CHANGE mode it is polled every
      loop pass so a change is detected promptly; in timed mode it is polled
      once per ``interval``.
    * ``on_run(client, args)`` — polled in the brick loop, at the same cadence
      as ``on_read``, unconditionally.
    """

    def __init__(self, name: str, **kwargs: Any) -> None:
        self.name = name
        self.on_read = kwargs.pop("on_read", None)
        self.on_write = kwargs.pop("on_write", None)
        self.on_run = kwargs.pop("on_run", None)
        self.interval = kwargs.pop("interval", ON_CHANGE)
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
        # Complex objects only: a sub-property changed and on_write is owed. The
        # per-leaf cloud frames are coalesced into a single on_write (fired once
        # with the whole object populated) by register() after seeding and by the
        # loop for live updates, instead of one callback per sub-property.
        self._on_write_pending = False
        self.last_poll = 0.0
        # Outbound (device → cloud) publish state, driven by pump() from the loop.
        self._dirty = False  # local value changed since the last publish (ON_CHANGE)
        self._last_push_ts = 0.0  # epoch secs of the last device→cloud publish
        self._has_pushed_once = False  # first publish is immediate (C++ !_has_been_updated_once)
        self._has_local_source = False  # the device has ever set this value (timed republish guard)
        self._last_warn_ts = 0.0  # epoch secs of the last "updated locally, not synced" warning

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

        # on_write is not polled: it fires immediately from the SSE handler.
        # Only the device→cloud callbacks are polled at ``interval``.
        self.runnable = any((self.on_run, self.on_read))

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

    def __setattr__(self, attr: str, value: Any) -> None:
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
        """Record a value set by the application; it is published by ``pump()``.

        This no longer publishes directly: the outbound value is sent by the
        loop's ``pump()`` according to the update policy (ON_CHANGE throttle or
        timed interval), mirroring the C++ library where a property write is
        published on the next ``update()`` pass that ``shouldBeUpdated()`` allows
        — not synchronously at assignment. Sync-time convergence pushes
        (``apply_missing`` / policy push-back) remain immediate.
        """
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
        self._has_local_source = True
        self._dirty = True  # pump() publishes it per policy (skipped while pending)

    def _do_push(self, now: float):
        """Publish the current value to the cloud now and update publish state.

        The single low-level send used by both the loop's ``pump()`` (policy
        driven) and the immediate sync-time convergence pushes. Clears the dirty
        flag and arms the ON_CHANGE throttle from ``now``.
        """
        if self._push is None:
            return
        self._push(self.name, self._value)
        self._last_push_ts = now
        self._has_pushed_once = True
        self._dirty = False

    def pump(self, now: float, interval: float):
        """Publish the local value to the cloud per the update policy.

        Called from the brick loop for each scalar leaf. ``interval`` is the
        owner object's policy selector (see ``ON_CHANGE`` / ``interval`` in the
        module docstring). No-op when there is no value to send.

        While no thing is assigned (``_pending``) the value cannot reach the
        cloud, so instead of publishing it warns on each local change (see
        ``_warn_local_only``) that the variable is being updated only locally.
        """
        if self._push is None or self._value is None:
            return
        if self._pending:
            self._warn_local_only(now)
            return
        if interval is None or interval < 0:
            # ON_CHANGE: publish a changed value, throttled to _MIN_PUBLISH_INTERVAL
            # (the very first publish is immediate, like C++ !_has_been_updated_once).
            if self._dirty and (not self._has_pushed_once or (now - self._last_push_ts) >= _MIN_PUBLISH_INTERVAL):
                self._do_push(now)
        else:
            # Timed (publishEvery): republish the current device value every
            # ``interval`` seconds, changed or not. Guarded on _has_local_source
            # so a purely cloud-controlled variable is never echoed back.
            if self._has_local_source and (now - self._last_push_ts) >= interval:
                self._do_push(now)

    def _warn_local_only(self, now: float):
        """Warn that a local change cannot be synced because no thing is assigned.

        Called from ``pump`` while ``_pending``. Fires only on a real local change
        (``_dirty``), throttled to ``_MIN_PUBLISH_INTERVAL`` like the ON_CHANGE
        publish so a fast ``on_read`` source does not flood the log — the same
        cadence as the HTTP 409 warning the daemon returns once a thing is
        assigned but the cloud is not steady. The dirty flag is then cleared: the
        current value is re-asserted to the cloud at sync time (``apply_cloud`` /
        ``apply_missing``) regardless of it, so nothing is lost.
        """
        if self._dirty and (now - self._last_warn_ts) >= _MIN_PUBLISH_INTERVAL:
            logger.warning(
                "ArduinoCloud: '%s' updated locally but NOT synced to the cloud — no thing "
                "assigned yet (cloud not steady); value kept local until a thing is available",
                self.name,
            )
            self._last_warn_ts = now
            self._dirty = False

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
            # divergence to avoid an echo loop. Immediate (convergence), not
            # subject to the outbound policy throttle.
            if self._value is not None and value != self._value:
                self._do_push(_now())
            return False

        if self.sync == MOST_RECENT_WINS and self._local_ts is not None and cloud_ts <= self._local_ts:
            # The local change is newer: keep it and push it up so the cloud
            # converges (guarded on divergence to avoid an echo loop).
            if self._value is not None and value != self._value:
                self._do_push(_now())
            return False

        if value == self._value:
            # Cloud already holds our value: nothing to publish.
            self._dirty = False
            return False
        self._value = value
        self._dirty = False  # cloud value adopted; discard any pending local push
        return True

    def apply_missing(self):
        """Resolve a ``lastvalue_missing`` sync frame: the cloud has no stored
        value for this variable, so the local value wins and is pushed up so the
        cloud converges. Applies to every sync policy. No-op if there is no local
        value to assert yet. Immediate (convergence), not throttled by the policy.
        """
        if self._value is not None:
            self._do_push(_now())

    # ── loop execution ─────────────────────────────────────────────────────────
    def run_sync(self, client):
        """Run this object's device→cloud callbacks once (called from the brick
        loop every ``interval`` seconds).

        Only ``on_run`` and ``on_read`` run here — they sample the device state
        into the local value (via ``set_local``); the actual device→cloud publish
        is done by ``pump()`` per the update policy. ``on_write`` is NOT fired
        here: a cloud→device update fires it immediately from the SSE handler
        (see ``fire_on_write``).
        """
        if self.on_run is not None:
            self.on_run(client, self.args)
        if self.on_read is not None:
            self.set_local(self.on_read(client))

    def fire_on_write(self, client):
        """Invoke this object's ``on_write`` callback with the current value.

        Called synchronously from the SSE handler the moment a cloud update is
        applied, mirroring the C++ ArduinoIoTCloud ``execCallbackOnChange`` /
        ``onUpdate`` dispatch, which runs as soon as the update message is
        decoded — not on a later, interval-gated pass.
        """
        if self.on_write is not None:
            self.on_write(client, self if self.is_complex else self._value)


# ── Re-exported complex objects (formerly from arduino_iot_cloud) ────────────


class Location(CloudObject):
    def __init__(self, name: str, **kwargs: Any) -> None:
        super().__init__(name, keys=("lat", "lon"), **kwargs)


class Color(CloudObject):
    def __init__(self, name: str, **kwargs: Any) -> None:
        super().__init__(name, keys=("hue", "sat", "bri"), **kwargs)


class ColoredLight(CloudObject):
    def __init__(self, name: str, **kwargs: Any) -> None:
        super().__init__(name, keys=("swi", "hue", "sat", "bri"), **kwargs)


class DimmedLight(CloudObject):
    def __init__(self, name: str, **kwargs: Any) -> None:
        super().__init__(name, keys=("swi", "bri"), **kwargs)


class Schedule(CloudObject):
    """A cloud schedule (frm/to/len/msk). Computes its active state in on_run."""

    def __init__(self, name: str, **kwargs: Any) -> None:
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
