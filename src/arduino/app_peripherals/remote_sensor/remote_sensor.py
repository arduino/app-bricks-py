# SPDX-FileCopyrightText: Copyright (C) 2025 ARDUINO SA <http://www.arduino.cc>
#
# SPDX-License-Identifier: MPL-2.0

import json
import os
import threading
import time
import asyncio
import numpy as np
from typing import Optional, Callable, Literal
from concurrent.futures import CancelledError, ThreadPoolExecutor, TimeoutError

import websockets

from .errors import RemoteSensorOpenError, RemoteSensorConfigError
from arduino.app_utils import Logger

logger = Logger("RemoteSensor")


class RemoteSensor:
    """
    RemoteSensor implementation that hosts a WebSocket server.

    This sensor acts as a WebSocket server that receives IoT telemetry data from connected clients.
    Only one client can be connected at a time.

    Clients can send data in JSON, CSV, or binary format. Each message is passed to the registered
    callback via the on_datapoint method.
    """

    def __init__(
        self,
        port: int = 8080,
        timeout: int = 3,
        data_format: Literal["json", "csv", "binary"] = "json",
        auto_reconnect: bool = True,
    ):
        """
        Initialize RemoteSensor WebSocket server.

        Args:
            port (int): Port to bind the server to (default: 8080)
            timeout (int): Connection timeout in seconds (default: 3)
            data_format (str): Expected data format from clients (default: "json")
                - "json": JSON object format. Callback receives the parsed dict directly.
                - "csv": CSV format with ",", "\\t", or " " as field separators and CRLF or LF
                  as line separator. Double quotes for escaping strings. Each line is a sensor
                  reading. Callback receives {"csv": "line_content"}.
                - "binary": Raw binary data. Callback receives {"binary": numpy_array}.
            auto_reconnect (bool, optional): Enable automatic reconnection on failure. Default: True.
        """
        self.protocol = "ws"
        self.port = port
        self.timeout = timeout
        if data_format not in ["json", "csv", "binary"]:
            raise RemoteSensorConfigError(f"Invalid data format: {data_format}. Must be 'json', 'csv', or 'binary'")
        self.data_format = data_format.lower()
        self.logger = logger
        self.name = self.__class__.__name__

        # Auto-reconnection parameters
        self.auto_reconnect = auto_reconnect
        self.auto_reconnect_delay = 1.0
        self.first_connection_max_retries = 10
    
        host_ip = os.getenv("HOST_IP")
        self._bind_ip = "0.0.0.0"
        self._external_ip = host_ip if host_ip is not None else self._bind_ip

        self._status: Literal["disconnected", "connected", "streaming", "paused"] = "disconnected"
        self._is_started = False
        self._server = None
        self._loop = None
        self._server_thread = None
        self._stop_event = asyncio.Event()
        self._client: Optional[websockets.ServerConnection] = None
        self._client_lock = asyncio.Lock()
        
        # Event handling
        # These callbacks doesn't require a lock as long as we're running on CPython
        self._on_datapoint_cb: Optional[Callable[[dict], None]] = None
        self._on_status_changed_cb: Callable[[str, dict], None] | None = None
        self._event_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="RemoteSensorCallbackRunner")

    @property
    def status(self) -> Literal["disconnected", "connected", "streaming", "paused"]:
        """Read-only property for camera status."""
        return self._status

    @property
    def url(self) -> str:
        """Return the WebSocket server address."""
        return f"{self.protocol}://{self._external_ip}:{self.port}"

    def start(self) -> None:
        """Start the WebSocket server."""
        self.logger.info("Starting remote sensor...")

        attempt = 0
        while not self.is_started():
            try:
                self._open_sensor()
                self._is_started = True
                self.logger.info(f"Successfully started {self.name}")
            except Exception as e:
                if not self.auto_reconnect:
                    raise
                attempt += 1
                if attempt >= self.first_connection_max_retries:
                    raise RemoteSensorOpenError(
                        f"Failed to start remote sensor after {self.first_connection_max_retries} attempts, last error is: {e}"
                    )

                delay = min(self.auto_reconnect_delay * (2 ** (attempt - 1)), 60)  # Exponential backoff
                self.logger.warning(
                    f"Failed to start remote sensor (attempt {attempt}/{self.first_connection_max_retries}). Retrying in {delay:.1f}s..."
                )
                time.sleep(delay)

    def _open_sensor(self) -> None:
        """Start the WebSocket server."""
        self._server_thread = threading.Thread(target=self._start_server_thread, daemon=True)
        self._server_thread.start()

        # Wait for server to start
        start_time = time.time()
        start_timeout = self.timeout
        while time.time() - start_time < start_timeout:
            if self._server is not None:
                self.logger.info(f"WebSocket remote sensor server started on {self.url}")
                return
            time.sleep(0.1)

        # Cleanup server thread if it failed to start in time
        if self._server_thread.is_alive():
            self._server_thread.join(timeout=1.0)

        raise RemoteSensorOpenError(f"Failed to start WebSocket server on {self.url}")

    def _start_server_thread(self) -> None:
        """Run WebSocket server in its own thread with event loop."""
        try:
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            self._loop.run_until_complete(self._start_server())
        except Exception as e:
            self.logger.error(f"WebSocket server thread error: {e}")
        finally:
            if self._loop and not self._loop.is_closed():
                self._loop.close()

    async def _start_server(self) -> None:
        """Start the WebSocket server."""
        try:
            self._stop_event.clear()

            self._server = await asyncio.wait_for(
                websockets.serve(
                    self._ws_handler,
                    self._bind_ip,
                    self.port,
                    open_timeout=self.timeout,
                    ping_timeout=self.timeout,
                    close_timeout=self.timeout,
                    ping_interval=20,
                ),
                timeout=self.timeout,
            )

            # Get the actual port if OS assigned one (i.e. when port=0)
            if self.port == 0:
                server_socket = list(self._server.sockets)[0]
                self.port = server_socket.getsockname()[1]

            await self._stop_event.wait()

        except TimeoutError as e:
            self.logger.error(f"Failed to start WebSocket server in a time ({self.timeout}s): {e}")
            raise
        except Exception as e:
            self.logger.error(f"Failed to start WebSocket server: {e}")
            raise
        finally:
            if self._server:
                self._server.close()
                await self._server.wait_closed()
                self._server = None

    async def _ws_handler(self, conn: websockets.ServerConnection) -> None:
        """Handle a connected WebSocket client. Only one client allowed at a time."""
        client_addr = f"{conn.remote_address[0]}:{conn.remote_address[1]}"

        async with self._client_lock:
            if self._client is not None:
                # Reject the new client
                self.logger.warning(f"Rejecting client {client_addr}: only one client allowed at a time")
                try:
                    await conn.send(json.dumps({"error": "Server busy", "message": "Only one client connection allowed at a time", "code": 1000}))
                    await conn.close(code=1000, reason="Server busy - only one client allowed")
                except Exception as e:
                    self.logger.warning(f"Error sending rejection message to {client_addr}: {e}")
                return

            # Accept the client
            self._client = conn

        self._set_status("connected", {"client_address": client_addr})

        self.logger.debug(f"Client connected: {client_addr}")

        try:
            # Send welcome message
            try:
                await self._send_to_client({
                    "status": "connected",
                    "message": "You are now connected to the RemoteSensor server",
                    "data_format": self.data_format,
                })
            except Exception as e:
                self.logger.warning(f"Could not send welcome message to {client_addr}: {e}")

            async for message in conn:
                datapoints = self._parse_message(message)
                if datapoints is not None and self._on_datapoint_cb is not None:
                    # Handle both single datapoint and list of datapoints
                    if not isinstance(datapoints, list):
                        datapoints = [datapoints]

                    for datapoint in datapoints:
                        try:
                            await self._loop.run_in_executor(None, self._on_datapoint_cb, datapoint)
                        except Exception as e:
                            self.logger.error(f"Error in datapoint callback: {e}")

        except websockets.exceptions.ConnectionClosed:
            self.logger.info(f"Client disconnected: {client_addr}")
        except Exception as e:
            self.logger.warning(f"Error handling client {client_addr}: {e}")
        finally:
            async with self._client_lock:
                if self._client == conn:
                    self._client = None
                    self._set_status("disconnected", {"client_address": client_addr})
                    self.logger.debug(f"Client disconnected: {client_addr}")

    def _parse_message(self, message):
        """
        Parse WebSocket message to extract datapoint(s) based on configured format.

        Returns:
            For json/binary: Single dict or None
            For csv: Single dict, list of dicts (if multiple lines), or None
        """
        try:
            if self.data_format == "json":
                # Parse JSON format
                if isinstance(message, bytes):
                    message = message.decode("utf-8")
                datapoint = json.loads(message)
                if not isinstance(datapoint, dict):
                    raise ValueError(f"Expected JSON object, got {type(datapoint)}")
                return datapoint

            elif self.data_format == "csv":
                # Parse CSV format
                if isinstance(message, bytes):
                    message = message.decode("utf-8")

                lines = message.replace("\r\n", "\n").split("\n")

                datapoints = []
                for line in lines:
                    line = line.strip()
                    if line:
                        datapoints.append({"csv": line})

                # Return list if multiple lines, single dict if one line, None if no lines
                if len(datapoints) == 0:
                    return None
                elif len(datapoints) == 1:
                    return datapoints[0]
                else:
                    return datapoints

            elif self.data_format == "binary":
                # Parse binary format
                if isinstance(message, str):
                    message = message.encode("utf-8")

                # Convert to numpy array
                data_array = np.frombuffer(message, dtype=np.uint8)
                return {"binary": data_array}

            else:
                logger.error(f"Unknown data format: {self.data_format}")
                return None

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON message: {e}")
            return None
        except Exception as e:
            logger.error(f"Error parsing message: {e}")
            return None

    def stop(self) -> None:
        """Stop the WebSocket server."""
        if not self.is_started():
            return
        
        self.logger.info("Stopping remote sensor...")

        try:
            self._close_sensor()
            self._event_executor.shutdown()
            self._is_started = False
            self.logger.info(f"Successfully stopped {self.name}")
        except Exception as e:
            self.logger.warning(f"Failed to stop remote sensor: {e}")

    def _close_sensor(self) -> None:
        """Stop the WebSocket server."""
        # Only attempt cleanup if the event loop is running
        if self._loop and not self._loop.is_closed() and self._loop.is_running():
            try:
                future = asyncio.run_coroutine_threadsafe(self._stop_and_disconnect_client(), self._loop)
                future.result(1.0)
            except CancelledError:
                self.logger.debug(f"Error stopping WebSocket server: CancelledError")
            except TimeoutError:
                self.logger.debug(f"Error stopping WebSocket server: TimeoutError")
            except Exception as e:
                self.logger.warning(f"Error stopping WebSocket server: {e}")

        # Wait for server thread to finish
        if self._server_thread and self._server_thread.is_alive():
            self._server_thread.join(timeout=10.0)

        # Reset state
        self._server = None
        self._loop = None
        self._client = None
        self._stop_event = None
        self._client_lock = None

    async def _stop_and_disconnect_client(self):
        """Set the async stop event and close the client connection."""
        # Send goodbye message and close the client connection
        if self._client:
            try:
                self.logger.debug("Disconnecting client...")
                # Send goodbye message before closing
                await self._send_to_client({
                    "status": "disconnecting",
                    "message": "Server is shutting down. Connection will be closed.",
                })
            except Exception as e:
                self.logger.warning(f"Error closing client in stop event: {e}")
            finally:
                if self._client:
                    await self._client.close()
                    self.logger.debug("Client connection closed")

        self._stop_event.set()

    async def _send_to_client(self, message: str | bytes | dict) -> None:
        """Send a message to the connected client."""
        if isinstance(message, dict):
            message = json.dumps(message)

        async with self._client_lock:
            if self._client is None:
                raise ConnectionError("No client connected to send message to")

            try:
                await self._client.send(message)
            except websockets.ConnectionClosedOK:
                self.logger.warning("Client has already closed the connection")
            except Exception:
                raise

    def on_datapoint(self, callback: Callable[[dict], None]) -> None:
        """
        Register a callback function to be called when a datapoint is received.

        The callback function will be called with a single argument: a dictionary containing
        the parsed data. The format of the dictionary depends on the data_format setting:

        - "json" format: The parsed JSON object is passed directly as a dict.
        - "csv" format: {"csv": "line_content"} where line_content is the CSV line string.
        - "binary" format: {"binary": numpy_array} where numpy_array contains the raw bytes as uint8.

        Args:
            callback (Callable): A function that takes a dict and returns None.
        """
        self._on_datapoint_cb = callback

    def is_started(self) -> bool:
        """Check if the sensor is started and running."""
        return self._is_started

    def on_status_changed(self, callback: Callable[[str, dict], None] | None):
        """Registers or removes a callback to be triggered on camera lifecycle events.

        When a camera status changes, the provided callback function will be invoked.
        If None is provided, the callback will be removed.

        Args:
            callback (Callable[[str, dict], None]): A callback that will be called every time the
                camera status changes with the new status and any associated data. The status names
                depend on the actual camera implementation being used. Some common events are:
                - 'connected': The camera has been reconnected.
                - 'disconnected': The camera has been disconnected.
                - 'streaming': The stream is streaming.
                - 'paused': The stream has been paused and is temporarily unavailable.
            callback (None): To unregister the current callback, if any.

        Example:
            def on_status(status: str, data: dict):
                print(f"Camera is now: {status}")
                print(f"Data: {data}")
                # Here you can add your code to react to the event

            camera.on_status_changed(on_status)
        """
        if callback is None:
            self._on_status_changed_cb = None
        else:

            def _callback_wrapper(new_status: str, data: dict):
                try:
                    callback(new_status, data)
                except Exception as e:
                    self.logger.error(f"Callback for '{new_status}' status failed with error: {e}")

            self._on_status_changed_cb = _callback_wrapper

    def _set_status(self, new_status: Literal["disconnected", "connected", "streaming", "paused"], data: dict | None = None) -> None:
        """
        Updates the current status of the camera and invokes the registered status
        changed callback in the background, if any.

        Only allowed states and transitions are considered, other states are ignored.
        Allowed states are:
            - disconnected
            - connected
            - streaming
            - paused

        Args:
            new_status (str): The name of the new status.
            data (dict): Additional data associated with the status change.
        """

        if self.status == new_status:
            return

        allowed_transitions = {
            "disconnected": ["connected"],
            "connected": ["disconnected", "streaming"],
            "streaming": ["paused", "disconnected"],
            "paused": ["streaming", "disconnected"],
        }

        # If new status is not in the state machine, ignore it
        if new_status not in allowed_transitions:
            return

        # Check if new_status is an allowed transition for the current status
        if new_status in allowed_transitions[self._status]:
            self._status = new_status
            if self._on_status_changed_cb is not None:
                self._event_executor.submit(self._on_status_changed_cb, new_status, data if data is not None else {})

    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()
        return False
