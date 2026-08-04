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
  local value: ``thing_unavailable`` (no thing assigned yet), ``lastvalue`` (the
  variable's stored cloud value, replayed with ``last_value: true``) or
  ``lastvalue_missing`` (thing assigned, no cloud value). Once the cloud reaches
  steady state a ``lastvalue``/``lastvalue_missing`` resync frame follows for
  clients that connected while unprovisioned. Every subsequent live change is an
  ``event: update``. Each event's JSON payload is ``{name, value, timestamp,
  last_value}`` (``thing_unavailable``/``lastvalue_missing`` carry only ``name``).
"""

import json
import threading
from datetime import datetime
from urllib.parse import quote, unquote, urlparse

import requests

from arduino.app_utils import Logger

from .unix_adapter import UnixHTTPAdapter

logger = Logger("ArduinoCloud")

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


def parse_timestamp(value) -> float | None:
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
        # If the URL uses the http+unix:// scheme, every session must mount the
        # UNIX-socket adapter; the socket path is the percent-encoded host part.
        parsed = urlparse(self._base)
        self._socket_path = unquote(parsed.netloc) if parsed.scheme == "http+unix" else None
        self._session = self._new_session()
        self._sse_sessions: list[requests.Session] = []
        self._sse_lock = threading.Lock()
        # Count of consecutive PUT failures, to distinguish a transient glitch
        # from a persistent stall in the daemon and to log a clear recovery.
        self._put_fail_count = 0

    def _new_session(self) -> requests.Session:
        session = requests.Session()
        if self._socket_path:
            session.mount("http+unix://", UnixHTTPAdapter(self._socket_path))
        return session

    def put_value(self, name: str, value) -> None:
        """Send a variable value to the daemon (best-effort; logs on failure).

        Failures are classified so the log points at the likely cause: a read
        timeout means the daemon accepted the request but never replied (it is
        stuck — e.g. blocked on the cloud broker); a connection error means the
        daemon/socket is unreachable. Consecutive failures are counted so a
        persistent stall is distinguishable from a one-off glitch, and a clear
        recovery line is logged once PUTs succeed again.
        """
        url = f"{self._base}/v1/variables/{quote(name, safe='')}"
        try:
            resp = self._session.put(url, json={"value": value}, timeout=_PUT_TIMEOUT)
        except requests.exceptions.ReadTimeout:
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
        except requests.exceptions.ConnectionError as e:
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
        except requests.RequestException as e:
            self._put_fail_count += 1
            logger.warning("ArduinoCloud: failed to send '%s' (consecutive failure #%d): %s", name, self._put_fail_count, e)
            return

        if resp.status_code == 409:
            # No thing assigned yet (cloud not steady): the daemon deliberately
            # did not queue the value. Expected during startup/reprovision — the
            # value is kept locally and pushed at sync time; log as a warning.
            logger.warning(
                "ArduinoCloud: '%s' not sent — no thing assigned yet (cloud not steady); value kept locally until sync",
                name,
            )
            return
        if resp.status_code >= 400:
            logger.warning("ArduinoCloud: PUT '%s' rejected by daemon: HTTP %s %s", name, resp.status_code, resp.text.strip())
            return
        if self._put_fail_count:
            logger.info("ArduinoCloud: '%s' delivered again after %d consecutive failure(s)", name, self._put_fail_count)
            self._put_fail_count = 0

    def stream_events(self, name: str, handler, stop_event: threading.Event, ready: threading.Event = None) -> None:
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
        url = f"{self._base}/v1/variables/{quote(name, safe='')}/events"
        session = self._new_session()
        with self._sse_lock:
            self._sse_sessions.append(session)

        backoff = 0.5
        while not stop_event.is_set():
            try:
                with session.get(url, stream=True, timeout=(_SSE_CONNECT_TIMEOUT, None)) as resp:
                    resp.raise_for_status()
                    backoff = 0.5  # reset after a successful connect
                    for event, payload in self._iter_events(resp, stop_event):
                        handler(event, payload)
                        if ready is not None and not ready.is_set():
                            ready.set()  # first frame delivered → unblock register's seed
            except Exception as e:  # noqa: BLE001 - reconnect on any transport error
                if stop_event.is_set():
                    break
                logger.debug("ArduinoCloud: SSE '%s' disconnected (%s); reconnecting in %.1fs", name, e, backoff)
            if stop_event.wait(backoff):
                break
            backoff = min(backoff * 2, _RECONNECT_MAX)

    @staticmethod
    def _iter_events(resp, stop_event: threading.Event):
        """Parse the SSE byte stream, yielding each complete event as
        ``(event_name, payload_dict)``. Stops when stop_event is set or the
        stream ends."""
        event = None
        data_lines: list[str] = []
        for raw in resp.iter_lines(decode_unicode=True):
            if stop_event.is_set():
                return
            if raw is None:
                continue
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
        """Close all sessions, unblocking any in-flight SSE reads."""
        with self._sse_lock:
            sessions = list(self._sse_sessions)
            self._sse_sessions.clear()
        for s in sessions:
            try:
                s.close()
            except Exception:  # noqa: BLE001
                pass
        try:
            self._session.close()
        except Exception:  # noqa: BLE001
            pass
