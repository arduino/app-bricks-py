# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

import os
import threading
import time
import warnings
from typing import Any
from urllib.parse import quote

from arduino.app_utils import brick, Logger

from .daemon_client import (
    DaemonClient,
    parse_timestamp,
    EVENT_LASTVALUE,
    EVENT_LASTVALUE_MISSING,
    EVENT_THING_UNAVAILABLE,
)
from .objects import CloudObject, CLOUD_WINS  # noqa: F401 (CLOUD_WINS re-exported)

logger = Logger("ArduinoCloud")

# The daemon serves its API on a UNIX socket (bind-mounted into the app
# container), so the brick talks to it over the socket by default — no network
# exposure. Override the whole URL with ARDUINO_CLOUD_CONNECTOR_URL (e.g. an
# http://127.0.0.1:5683 endpoint on the host), or just the socket path with
# ARDUINO_CLOUD_CONNECTOR_SOCKET.
_DEFAULT_DAEMON_SOCKET = "/run/arduino-cloud-connector/daemon.sock"
_LOOP_INTERVAL = 0.1  # seconds between callback-poll passes
# How long register() waits for a leaf's first (sync) frame before returning
# without a synchronous seed (the stream keeps retrying and seeds on connect).
_SEED_TIMEOUT = 10.0
# How often the loop re-checks that every registered leaf still has a live SSE
# listener and re-subscribes any that is missing (e.g. a subscribe that failed
# at register time, or a listener thread that died).
_SUB_CHECK_INTERVAL = 5.0

# Sentinel for the deprecated constructor arguments: lets us tell "not passed"
# apart from a real value (so the common ArduinoCloud() call stays silent).
_DEPRECATED = object()


@brick
class ArduinoCloud:
    """Arduino Cloud client for exchanging variables with the Arduino Cloud daemon.

    Connectivity, provisioning and the cloud handshake are owned by the
    ``arduino-cloud-connector`` daemon running on the board; this brick exchanges
    variable values with it over its localhost REST/SSE API. The public
    interface (constructor, ``register``, attribute get/set, the ``on_write`` /
    ``on_read`` / ``on_run`` callbacks and the re-exported ``Location`` /
    ``Color`` / ``ColoredLight`` / ``DimmedLight`` / ``Schedule`` objects) is
    unchanged from the previous ``arduino_iot_cloud``-based implementation.

    Per-variable conflict resolution is selectable via the ``sync`` argument to
    ``register`` (``DEVICE_WINS`` / ``CLOUD_WINS`` / ``MOST_RECENT_WINS``,
    default ``CLOUD_WINS``).
    """

    def __init__(
        self,
        device_id: str = _DEPRECATED,
        secret: str = _DEPRECATED,
        server: str = _DEPRECATED,
        port: int = _DEPRECATED,
        daemon_url: str = None,
    ) -> None:
        """Initialize the Arduino Cloud client.

        Args:
            device_id (str): Deprecated and ignored. The daemon owns the device
                             identity and provisioning.
            secret (str): Deprecated and ignored (see device_id).
            server (str): Deprecated and ignored. The daemon connects to the
                          cloud broker on the brick's behalf.
            port (int): Deprecated and ignored (see server).
            daemon_url (str, optional): Base URL of the local daemon REST API.
                If omitted, uses the ARDUINO_CLOUD_CONNECTOR_URL environment
                variable, otherwise the daemon's UNIX socket
                (http+unix://<ARDUINO_CLOUD_CONNECTOR_SOCKET or
                /run/arduino-cloud-connector/daemon.sock>).
        """
        legacy = {"device_id": device_id, "secret": secret, "server": server, "port": port}
        if passed := [name for name, value in legacy.items() if value is not _DEPRECATED]:
            warnings.warn(
                f"ArduinoCloud argument(s) {passed} are deprecated and ignored: device "
                "identity, credentials and broker connectivity are now managed by the "
                "arduino-cloud-connector daemon. Pass daemon_url to reach a non-default daemon.",
                DeprecationWarning,
                stacklevel=2,
            )

        url = daemon_url or self._default_daemon_url()
        self._client = DaemonClient(url)
        self._records: dict[str, CloudObject] = {}
        self._lock = threading.RLock()
        self._stop = threading.Event()
        # One SSE listener thread per scalar leaf, keyed by leaf name, so the
        # loop can verify each leaf is still subscribed and re-subscribe if not.
        self._listeners: dict[str, threading.Thread] = {}
        self._last_sub_check = 0.0
        self._started = False

    @staticmethod
    def _default_daemon_url() -> str:
        if url := os.getenv("ARDUINO_CLOUD_CONNECTOR_URL"):
            return url
        socket_path = os.getenv("ARDUINO_CLOUD_CONNECTOR_SOCKET", _DEFAULT_DAEMON_SOCKET)
        return "http+unix://" + quote(socket_path, safe="")

    # ── lifecycle (managed by the App framework) ────────────────────────────────
    def start(self):
        """Mark the brick started and ensure every registered leaf is subscribed.

        Subscription and the synchronous initial seed already happen in
        ``register``; this is a safety net that (re)subscribes anything not yet
        listening — e.g. a leaf whose register-time subscribe failed.
        """
        with self._lock:
            self._started = True
        self._ensure_subscribed()

    def loop(self):
        """Sample device→cloud callbacks (on_run / on_read) and publish per policy.

        Each pass, for every registered object: run its poll callbacks when due
        (every pass in ON_CHANGE mode, once per ``interval`` in timed mode), then
        publish each scalar leaf via ``pump()`` (ON_CHANGE throttle or timed
        republish). Scalar on_write fires immediately from the SSE handler; a
        complex object's coalesced on_write (one call for the whole object) is
        delivered here, once per pass.
        """
        now = time.time()
        # Periodically make sure every registered leaf still has a live SSE
        # listener (recovers a failed subscribe or a dead listener thread).
        if now - self._last_sub_check >= _SUB_CHECK_INTERVAL:
            self._last_sub_check = now
            self._ensure_subscribed()
        with self._lock:
            records = list(self._records.values())
        for record in records:
            try:
                if record.runnable and self._poll_due(now, record):
                    record.run_sync(self)
                    record.last_poll = now
                for leaf in record.leaves():
                    leaf.pump(now, record.interval)
                # Deliver a complex object's coalesced on_write once per pass,
                # with the whole object populated (see _make_handler). Scalars
                # fire immediately from the handler, so skip them here.
                if record.is_complex:
                    self._fire_pending_on_write(record)
            except Exception as e:
                logger.exception(f"Callback error for '{record.name}': {e}")
        time.sleep(_LOOP_INTERVAL)

    def _fire_pending_on_write(self, record: CloudObject):
        """Fire a complex object's coalesced on_write once, if one is owed.

        Checks and clears the pending flag under the lock, then invokes the
        callback outside it (user callbacks must not hold the state lock). Scalar
        objects never set the flag — they fire on_write immediately from the SSE
        handler — so this is a no-op for them.
        """
        fire = False
        with self._lock:
            if record._on_write_pending:
                record._on_write_pending = False
                fire = True
        if fire:
            record.fire_on_write(self)

    def stop(self):
        """Stop the brick and tear down the SSE listener threads."""
        with self._lock:
            threads = list(self._listeners.values())
        logger.info("ArduinoCloud: stopping — closing sessions and joining %d listener(s)", len(threads))
        self._stop.set()
        self._client.close()
        for thread in threads:
            thread.join(timeout=2)

    @staticmethod
    def _poll_due(now: float, record: CloudObject) -> bool:
        # In ON_CHANGE mode (negative interval) this is always true, so on_read/
        # on_run are sampled every loop pass; in timed mode they poll once per
        # interval. The actual cloud publish is throttled/timed in leaf.pump().
        return record.last_poll == 0.0 or (now - record.last_poll) >= record.interval

    # ── registration ─────────────────────────────────────────────────────────
    def register(self, aiotobj: str | Any, **kwargs: Any):
        """Register a variable or object with the Arduino Cloud client.

        Args:
            aiotobj (str | Any): The variable name, or a cloud object
                                 (Location/Color/ColoredLight/DimmedLight/
                                 Schedule) to register.
            **kwargs (Any): value, on_read, on_write, on_run, args, sync
                            (DEVICE_WINS/CLOUD_WINS/MOST_RECENT_WINS) and interval.
                            ``interval`` selects the device→cloud update policy,
                            like the C++ ``addProperty`` seconds argument:
                            ``ON_CHANGE`` (default) publishes changes throttled to
                            ~0.5s; a positive value publishes the current value
                            every N seconds (timed).
        """
        if isinstance(aiotobj, str):
            aiotobj = CloudObject(aiotobj, **kwargs)
        elif kwargs:
            raise TypeError("kwargs are not allowed when registering a cloud object instance")

        aiotobj.bind(self._client.put_value)
        with self._lock:
            self._records[aiotobj.name] = aiotobj

        # Open one SSE stream per leaf and wait for its first (sync) frame, so the
        # value is resolved per policy before register() returns; that same stream
        # then stays open for live updates — no second connection re-announces the
        # last value. If the daemon is momentarily unreachable the wait times out
        # and the stream seeds later, on connect.
        for leaf in aiotobj.leaves():
            self._subscribe_leaf(leaf, wait_ready=True)

        # A complex object's leaves seed as separate frames; deliver a single
        # on_write with the fully-seeded object rather than one per sub-property.
        self._fire_pending_on_write(aiotobj)

    def get(self, name: str, default: Any = None) -> Any:
        """Return a registered variable's value, or default if unset/unknown."""
        with self._lock:
            record = self._records.get(name)
        if record is None:
            return default
        value = record.value
        return default if value is None else value

    # ── SSE subscription ───────────────────────────────────────────────────────
    def _subscribe_leaf(self, leaf: CloudObject, wait_ready: bool = False):
        """Start (once) the single SSE listener thread for a scalar leaf.

        The listener's first frame seeds the leaf (resolving the local value per
        policy); the same open stream then carries live updates — so the last
        value is delivered on one connection, never re-announced by a second one.

        With ``wait_ready`` (register), block until that first frame has been
        applied, but no longer than ``_SEED_TIMEOUT`` so a momentarily
        unreachable daemon does not hang register — the stream keeps retrying and
        seeds when it connects.

        No-op if the leaf already has a live listener (so the periodic
        re-subscription check does not start duplicates).
        """
        with self._lock:
            existing = self._listeners.get(leaf.name)
            if existing is not None and existing.is_alive():
                return

        ready = threading.Event() if wait_ready else None
        thread = threading.Thread(
            target=self._client.stream_events,
            args=(leaf.name, self._make_handler(leaf), self._stop, ready),
            name=f"ArduinoCloud.sse.{leaf.name}",
            daemon=True,
        )
        thread.start()
        with self._lock:
            self._listeners[leaf.name] = thread

        if ready is not None and not ready.wait(timeout=_SEED_TIMEOUT):
            logger.warning(
                "ArduinoCloud: '%s' not seeded within %.0fs (daemon slow/unreachable); keeping local value — the stream will seed when it connects",
                leaf.name,
                _SEED_TIMEOUT,
            )

    def _ensure_subscribed(self):
        """(Re)subscribe any registered leaf whose SSE listener is missing or dead.

        Normally every leaf is subscribed in ``register``; this recovers a leaf
        whose subscribe failed (e.g. a transient error at register time) or whose
        listener thread died, so no property is ever left without a live stream.
        """
        with self._lock:
            records = list(self._records.values())
        for record in records:
            for leaf in record.leaves():
                with self._lock:
                    thread = self._listeners.get(leaf.name)
                    alive = thread is not None and thread.is_alive()
                if not alive:
                    logger.warning("ArduinoCloud: '%s' has no live SSE listener; re-subscribing", leaf.name)
                    self._subscribe_leaf(leaf)

    def _make_handler(self, leaf: CloudObject):
        """Build the SSE event handler for a leaf.

        Dispatches on the event name (see daemon_client): the sync frames
        (thing_unavailable / lastvalue / lastvalue_missing) resolve the leaf's
        local value and move it in/out of the pending state; live ``update``
        events apply cloud changes (and are ignored while pending, since only a
        sync frame ends the "no thing assigned" state). Whenever an applied cloud
        value wins and actually changes the local value (``apply_cloud`` returns
        True) on_write is delivered: for a scalar variable it fires immediately,
        as soon as the message arrives (C++ ArduinoIoTCloud synchronous onUpdate
        parity). For a complex object each sub-property arrives as its own frame,
        so on_write is coalesced — the owner is flagged and fired once, with the
        whole object populated, by register() after the initial seeding and by
        the loop for live updates — rather than once per sub-property.

        on_write is fired outside the lock (like the loop's run_sync, which also
        runs callbacks unlocked): user callbacks must not block the state lock
        held by other listeners and the poll loop.
        """

        def handle(event: str, payload: dict):
            owner_to_fire = None
            with self._lock:
                if event == EVENT_THING_UNAVAILABLE:
                    if not leaf._pending:
                        leaf._pending = True
                        logger.warning(
                            "ArduinoCloud: '%s' — no thing assigned yet; keeping local value, will sync when the thing becomes available",
                            leaf.name,
                        )
                    return

                if event == EVENT_LASTVALUE_MISSING:
                    leaf._pending = False
                    logger.debug("ArduinoCloud: '%s' has no cloud value; local value wins", leaf.name)
                    leaf.apply_missing()
                    return

                if event == EVENT_LASTVALUE:
                    leaf._pending = False
                    value = payload.get("value")
                    ts = parse_timestamp(payload.get("timestamp"))
                    logger.debug("ArduinoCloud: '%s' sync lastvalue=%r ts=%s", leaf.name, value, ts)
                    changed = leaf.apply_cloud(value, ts)
                else:
                    # Live update. Ignore while pending: only a sync frame ends
                    # the "no thing assigned" state.
                    if leaf._pending:
                        return
                    value = payload.get("value")
                    ts = parse_timestamp(payload.get("timestamp"))
                    logger.debug("ArduinoCloud: cloud update for '%s': value=%r ts=%s", leaf.name, value, ts)
                    changed = leaf.apply_cloud(value, ts)

                if changed:
                    owner = leaf._owner
                    if owner.is_complex:
                        # Coalesce: the sub-properties of one complex object
                        # arrive as separate per-leaf frames (initial sync or a
                        # multi-attribute cloud change). Flag the owner and let
                        # register() (after seeding) or the loop fire on_write
                        # once, with the whole object populated — not once per
                        # sub-property.
                        owner._on_write_pending = True
                    else:
                        owner_to_fire = owner

            # Scalars fire on_write immediately (outside the lock) so a
            # cloud→device update invokes the callback the moment the message
            # arrives (C++ parity). Complex objects are coalesced and fired once
            # by register()/the loop (see above).
            if owner_to_fire is not None:
                owner_to_fire.fire_on_write(self)

        return handle

    # ── attribute-style variable access ─────────────────────────────────────────
    def __getattr__(self, name: str):
        """Intercept access to cloud variables as natural attributes."""
        records = self.__dict__.get("_records")
        if records is not None and name in records:
            record = records[name]
            return record if record.is_complex else record.value
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def __setattr__(self, name: str, value: Any) -> None:
        """Intercept assignment to cloud variables as natural attributes."""
        if name.startswith("_"):
            super().__setattr__(name, value)
            return
        records = self.__dict__.get("_records")
        if records is not None and name in records:
            with self._lock:
                records[name].set_local(value)
            return
        super().__setattr__(name, value)
