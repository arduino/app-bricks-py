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
    ):
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
        self._listeners: list[threading.Thread] = []
        self._started = False

    @staticmethod
    def _default_daemon_url() -> str:
        if url := os.getenv("ARDUINO_CLOUD_CONNECTOR_URL"):
            return url
        socket_path = os.getenv("ARDUINO_CLOUD_CONNECTOR_SOCKET", _DEFAULT_DAEMON_SOCKET)
        return "http+unix://" + quote(socket_path, safe="")

    # ── lifecycle (managed by the App framework) ────────────────────────────────
    def start(self):
        """Start the brick: subscribe to every already-registered variable."""
        with self._lock:
            self._started = True
            for record in self._records.values():
                self._subscribe(record)

    def loop(self):
        """Poll registered objects' callbacks (on_run / on_read / on_write)."""
        now = time.time()
        with self._lock:
            records = list(self._records.values())
        for record in records:
            try:
                if record.runnable and self._poll_due(now, record):
                    record.run_sync(self)
                    record.last_poll = now
            except Exception as e:
                logger.exception(f"Callback error for '{record.name}': {e}")
        time.sleep(_LOOP_INTERVAL)

    def stop(self):
        """Stop the brick and tear down the SSE listener threads."""
        logger.info("ArduinoCloud: stopping — closing sessions and joining %d listener(s)", len(self._listeners))
        self._stop.set()
        self._client.close()
        for thread in self._listeners:
            thread.join(timeout=2)

    @staticmethod
    def _poll_due(now: float, record: CloudObject) -> bool:
        return record.last_poll == 0.0 or (now - record.last_poll) >= record.interval

    # ── registration ─────────────────────────────────────────────────────────
    def register(self, aiotobj: str | Any, **kwargs: Any):
        """Register a variable or object with the Arduino Cloud client.

        Args:
            aiotobj (str | Any): The variable name, or a cloud object
                                 (Location/Color/ColoredLight/DimmedLight/
                                 Schedule) to register.
            **kwargs (Any): value, on_read, on_write, on_run, interval, args and
                            sync (DEVICE_WINS/CLOUD_WINS/MOST_RECENT_WINS).
        """
        if isinstance(aiotobj, str):
            aiotobj = CloudObject(aiotobj, **kwargs)
        elif kwargs:
            raise TypeError("kwargs are not allowed when registering a cloud object instance")

        aiotobj.bind(self._client.put_value)
        with self._lock:
            self._records[aiotobj.name] = aiotobj

        # Synchronous initial sync: contact the daemon for each leaf's first
        # frame and resolve the local value per sync policy before returning, so
        # the variable holds the right value as soon as register() returns.
        for leaf in aiotobj.leaves():
            self._seed_leaf(leaf)

        with self._lock:
            if self._started:
                self._subscribe(aiotobj)

    def get(self, name: str, default: Any = None) -> Any:
        """Return a registered variable's value, or default if unset/unknown."""
        with self._lock:
            record = self._records.get(name)
        if record is None:
            return default
        value = record.value
        return default if value is None else value

    # ── SSE subscription ───────────────────────────────────────────────────────
    def _seed_leaf(self, leaf: CloudObject):
        """Synchronously fetch and apply a leaf's first (sync) frame from the
        daemon, resolving the local value per policy before register() returns.

        Logs and returns on a connection error; the streaming thread (started in
        start()) will retry and re-seed on connect, so a momentarily unreachable
        daemon does not lose the variable.
        """
        handler = self._make_handler(leaf)
        try:
            evt = self._client.fetch_initial(leaf.name, self._stop)
        except Exception as e:  # noqa: BLE001 - any transport error is non-fatal here
            logger.error(
                "ArduinoCloud: initial sync for '%s' failed: %s — keeping local value; will retry when the stream connects",
                leaf.name,
                e,
            )
            return
        if evt is not None:
            handler(*evt)

    def _subscribe(self, record: CloudObject):
        """Start one SSE listener thread per scalar leaf of the record."""
        for leaf in record.leaves():
            thread = threading.Thread(
                target=self._client.stream_events,
                args=(leaf.name, self._make_handler(leaf), self._stop),
                name=f"ArduinoCloud.sse.{leaf.name}",
                daemon=True,
            )
            thread.start()
            self._listeners.append(thread)

    def _make_handler(self, leaf: CloudObject):
        """Build the SSE event handler for a leaf.

        Dispatches on the event name (see daemon_client): the sync frames
        (thing_unavailable / lastvalue / lastvalue_missing) resolve the leaf's
        local value and move it in/out of the pending state; live ``update``
        events apply cloud changes (and are ignored while pending, since only a
        sync frame ends the "no thing assigned" state). A ``lastvalue`` whose
        cloud value wins and changes the local value fires on_write too, matching
        the C++ ArduinoIoTCloud sync semantics (initial sync and reconnect resync).
        """

        def handle(event: str, payload: dict):
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
                    # Resolve the synced cloud value per policy. Mirroring the C++
                    # ArduinoIoTCloud library (execCallbackOnSync -> onUpdate), fire
                    # on_write whenever the cloud value wins and actually changes
                    # the local value (apply_cloud returns True) — on the initial
                    # sync and on a reconnect resync alike, so an actuator is
                    # restored to a cloud state changed while offline.
                    if leaf.apply_cloud(value, ts):
                        leaf._owner.on_write_scheduled = True
                    return

                # Live update. Ignore while pending: only a sync frame ends the
                # "no thing assigned" state.
                if leaf._pending:
                    return
                value = payload.get("value")
                ts = parse_timestamp(payload.get("timestamp"))
                logger.debug("ArduinoCloud: cloud update for '%s': value=%r ts=%s", leaf.name, value, ts)
                if leaf.apply_cloud(value, ts):
                    leaf._owner.on_write_scheduled = True

        return handle

    # ── attribute-style variable access ─────────────────────────────────────────
    def __getattr__(self, name: str):
        """Intercept access to cloud variables as natural attributes."""
        records = self.__dict__.get("_records")
        if records is not None and name in records:
            record = records[name]
            return record if record.is_complex else record.value
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def __setattr__(self, name: str, value: Any):
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
