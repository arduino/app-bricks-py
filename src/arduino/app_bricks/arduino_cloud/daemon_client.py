# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""HTTP transport to the arduino-cloud-connector daemon's localhost REST/SSE API.

The daemon owns the MQTT connection to Arduino Cloud; the brick only exchanges
variable values with it over two endpoints (RFC-13 §8):

* ``PUT /v1/variables/{name}`` with body ``{"value": <any>}`` — queue a value
  for ordered delivery to the cloud.
* ``GET /v1/variables/{name}/events`` — a Server-Sent Events stream. The first
  event is always one of three "sync" events telling the client how to seed its
  local value: ``thing_unavailable`` (thing not reachable: the board has no
  internet connectivity, has not been provisioned or has no thing assigned),
  ``lastvalue`` (the variable's stored cloud value, replayed with
  ``last_value: true``) or ``lastvalue_missing`` (thing reachable, no cloud
  value). Once the cloud reaches steady state a ``lastvalue``/``lastvalue_missing``
  resync frame follows for clients that connected while the thing was not
  reachable. Every subsequent live change is an
  ``event: update``. Each event's JSON payload is ``{name, value, timestamp,
  last_value}`` (``thing_unavailable``/``lastvalue_missing`` carry only ``name``).

The daemon URL forms are described in ``unix_adapter``.
"""

import http.client
import json
import socket
import threading
import uuid
from datetime import datetime
from urllib.parse import quote

from collections.abc import Callable, Generator, Iterable, Iterator
from contextlib import contextmanager
from typing import Any

from arduino.app_utils import Logger

from .unix_adapter import HTTPEndpoint

logger = Logger("ArduinoCloud")

# Identifies this app instance to the daemon, sent on BOTH the value PUT and
# the SSE subscription. It is what lets the daemon avoid streaming an app its
# own write back: without it the echo of a PUT arrives after the app has
# already computed its next value, and a CLOUD_WINS variable adopts the stale
# echoed value — so a read-modify-write app (counter = counter + 1) silently
# loses an increment. Values coming from the cloud are never filtered.
CLIENT_ID_HEADER = "X-App-Client-ID"

_PUT_TIMEOUT = 10.0  # seconds for a value PUT
_SSE_CONNECT_TIMEOUT = 10.0  # seconds to establish the SSE connection
_RECONNECT_MAX = 5.0  # max backoff between SSE reconnects

# SSE event names exchanged with the daemon (see its internal/variables package).
# The first frame of every stream is always one of the three "sync" events;
# subsequent live changes are EVENT_UPDATE.
EVENT_UPDATE = "update"
EVENT_LASTVALUE = "lastvalue"
EVENT_LASTVALUE_MISSING = "lastvalue_missing"
EVENT_THING_UNAVAILABLE = "thing_unavailable"


def parse_timestamp(value: object) -> float | None:
    """Parse an RFC3339 timestamp from the daemon into epoch seconds.

    The daemon emits Go RFC3339Nano (e.g. ``2026-06-22T10:00:00.123456789Z``),
    whose fractional part can exceed datetime's 6-digit limit, so it is
    truncated to microseconds. Returns None if the value is missing/unparseable.
    """
    if not value or not isinstance(value, str):
        return None
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    # Truncate an over-long fractional second component to 6 digits.
    if "." in text:
        head, _, tail = text.partition(".")
        frac = ""
        rest = ""
        for i, ch in enumerate(tail):
            if ch.isdigit():
                frac += ch
            else:
                rest = tail[i:]
                break
        text = f"{head}.{frac[:6]}{rest}"
    try:
        return datetime.fromisoformat(text).timestamp()
    except ValueError:
        logger.debug("ArduinoCloud: could not parse timestamp %r", value)
        return None


class DaemonClient:
    """Thin client for the daemon REST/SSE API."""

    def __init__(self, base_url: str) -> None:
        self._base = base_url.rstrip("/")
        self._endpoint = HTTPEndpoint(self._base)
        # One identity per app instance, generated here and never configurable:
        # if two apps could be made to share it they would silently stop seeing
        # each other's writes. Being per-app rather than per-stream also means
        # it survives an SSE reconnect, so a PUT in flight across a reconnect
        # still matches. Sent on every request, streams included.
        self._client_id = str(uuid.uuid4())
        self._headers = {CLIENT_ID_HEADER: self._client_id}
        # Open SSE connections, shut down by close() to unblock their reads
        self._streams: set[socket.socket] = set()
        self._streams_lock = threading.Lock()
        self._closed = threading.Event()
        # Count of consecutive PUT failures, to distinguish a transient glitch
        # from a persistent stall in the daemon and to log a clear recovery.
        self._put_fail_count = 0

    def _put(self, path: str, value: object) -> tuple[int, str]:
        """Send a value PUT and return the response status and body."""
        body = json.dumps({"value": value}, allow_nan=False)
        conn = self._endpoint.connection(_PUT_TIMEOUT)
        try:
            try:
                conn.connect()
            except OSError as e:
                # Report a connect timeout as unreachable, not as a stalled daemon
                raise ConnectionError(str(e)) from e
            conn.request("PUT", path, body=body, headers={**self._headers, "Content-Type": "application/json"})
            resp = conn.getresponse()
            return resp.status, resp.read().decode("utf-8", errors="replace")
        finally:
            conn.close()

    def put_value(self, name: str, value: object) -> None:
        """Send a variable value to the daemon (best-effort; logs on failure).

        Failures are classified so the log points at the likely cause: a read
        timeout means the daemon accepted the request but never replied (it is
        stuck — e.g. blocked on the cloud broker); a connection error means the
        daemon/socket is unreachable. Consecutive failures are counted so a
        persistent stall is distinguishable from a one-off glitch, and a clear
        recovery line is logged once PUTs succeed again.
        """
        path = f"{self._endpoint.path_prefix}/v1/variables/{quote(name, safe='')}"
        try:
            status, text = self._put(path, value)
        except TimeoutError:
            self._put_fail_count += 1
            logger.warning(
                "ArduinoCloud: PUT '%s' timed out after %.0fs (consecutive failure #%d) — "
                "the daemon accepted the request but never responded, so it is likely "
                "blocked (e.g. wedged on the cloud broker connection). Values are NOT "
                "reaching the cloud; restarting arduino-cloud-connector clears it.",
                name,
                _PUT_TIMEOUT,
                self._put_fail_count,
            )
            return
        except ConnectionError as e:
            self._put_fail_count += 1
            logger.warning(
                "ArduinoCloud: cannot reach the daemon at %s to send '%s' (consecutive "
                "failure #%d): %s — is arduino-cloud-connector running and its socket mounted?",
                self._base,
                name,
                self._put_fail_count,
                e,
            )
            return
        except (OSError, http.client.HTTPException, ValueError) as e:
            self._put_fail_count += 1
            logger.warning("ArduinoCloud: failed to send '%s' (consecutive failure #%d): %s", name, self._put_fail_count, e)
            return

        if status == 409:
            # Thing not reachable (cloud not steady: the board has no internet
            # connectivity, has not been provisioned or has no thing assigned):
            # the daemon deliberately did not queue the value. Expected during startup/reprovision — the
            # value is kept locally and pushed at sync time; log as a warning.
            logger.warning(
                "ArduinoCloud: '%s' not sent — no thing assigned yet (cloud not steady); value kept locally until sync",
                name,
            )
            return
        if status >= 400:
            logger.warning("ArduinoCloud: PUT '%s' rejected by daemon: HTTP %s %s", name, status, text.strip())
            return
        if self._put_fail_count:
            logger.info("ArduinoCloud: '%s' delivered again after %d consecutive failure(s)", name, self._put_fail_count)
            self._put_fail_count = 0

    def stream_events(
        self, name: str, handler: Callable[[str, dict[str, Any]], None], stop_event: threading.Event, ready: threading.Event | None = None
    ) -> None:
        """Stream SSE events for a variable until stop_event is set.

        ``handler(event_name, payload_dict)`` is called for each event. The first
        frame the daemon sends on every (re)connection is a sync frame
        (``thing_unavailable`` / ``lastvalue`` / ``lastvalue_missing``); the rest
        are live ``update`` events. The stream is reconnected with capped
        exponential backoff on any error, so the sync frame is replayed and no
        state is lost across reconnects.

        If ``ready`` is given it is set as soon as the first frame has been
        delivered to the handler. ``register`` waits on it for a synchronous
        initial seed and then lets this same connection carry on with live
        updates — so the last value is delivered on a single stream, not
        re-announced by a second connection.
        """
        path = f"{self._endpoint.path_prefix}/v1/variables/{quote(name, safe='')}/events"
        backoff = 0.5
        while not (stop_event.is_set() or self._closed.is_set()):
            try:
                with self._subscribe(path) as resp:
                    backoff = 0.5  # reset after a successful connect
                    for event, payload in self._iter_events(resp, stop_event):
                        handler(event, payload)
                        if ready is not None and not ready.is_set():
                            ready.set()  # first frame delivered → unblock register's seed
            except Exception as e:  # noqa: BLE001 - reconnect on any transport error
                if stop_event.is_set() or self._closed.is_set():
                    break
                logger.debug("ArduinoCloud: SSE '%s' disconnected (%s); reconnecting in %.1fs", name, e, backoff)
            if stop_event.wait(backoff):
                break
            backoff = min(backoff * 2, _RECONNECT_MAX)

    @contextmanager
    def _subscribe(self, path: str) -> Generator[http.client.HTTPResponse]:
        """Open an SSE stream that close() can interrupt."""
        conn = self._endpoint.connection(_SSE_CONNECT_TIMEOUT)
        try:
            conn.connect()
            sock = conn.sock
            # Frames can be minutes apart and the daemon sends no heartbeat
            sock.settimeout(None)
            with self._streams_lock:
                if self._closed.is_set():
                    raise ConnectionAbortedError("daemon client closed")
                self._streams.add(sock)
            try:
                conn.request("GET", path, headers=self._headers)
                with conn.getresponse() as resp:
                    if resp.status != 200:
                        raise http.client.HTTPException(f"HTTP {resp.status} {resp.reason}")
                    yield resp
            finally:
                with self._streams_lock:
                    self._streams.discard(sock)
        finally:
            conn.close()

    @staticmethod
    def _iter_events(lines: Iterable[bytes], stop_event: threading.Event) -> Iterator[tuple[str, dict[str, Any]]]:
        """Parse the SSE byte stream, yielding each complete event as
        ``(event_name, payload_dict)``. Stops when stop_event is set or the
        stream ends."""
        event = None
        data_lines: list[str] = []
        for line in lines:
            if stop_event.is_set():
                return
            raw = line.decode("utf-8", errors="replace").rstrip("\r\n")
            if raw == "":  # blank line terminates an event
                if data_lines:
                    try:
                        payload = json.loads("\n".join(data_lines))
                    except ValueError:
                        payload = None
                    if isinstance(payload, dict):
                        yield (event or "message", payload)
                event = None
                data_lines = []
                continue
            if raw.startswith(":"):  # comment / heartbeat
                continue
            field, _, val = raw.partition(":")
            if val.startswith(" "):
                val = val[1:]
            if field == "event":
                event = val
            elif field == "data":
                data_lines.append(val)

    def close(self) -> None:
        """Shut down every open stream, unblocking its read; streams do not reconnect afterwards."""
        with self._streams_lock:
            self._closed.set()
            for sock in self._streams:
                try:
                    sock.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass  # already closed by its stream
