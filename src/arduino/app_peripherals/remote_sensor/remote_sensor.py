# SPDX-FileCopyrightText: Copyright (C) 2025 ARDUINO SA <http://www.arduino.cc>
#
# SPDX-License-Identifier: MPL-2.0

import json
import threading
import time
import asyncio
import numpy as np
from typing import Optional, Callable, Dict, Any, Literal
from concurrent.futures import CancelledError, TimeoutError

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
        host: str = "0.0.0.0",
        port: int = 8080,
        timeout: int = 10,
        data_format: Literal["json", "csv", "binary"] = "json",
    ):
        """
        Initialize RemoteSensor WebSocket server.

        Args:
            host (str): Host address to bind the server to (default: "0.0.0.0")
            port (int): Port to bind the server to (default: 8080)
            timeout (int): Connection timeout in seconds (default: 10)
            data_format (str): Expected data format from clients (default: "json")
                - "json": JSON object format. Callback receives the parsed dict directly.
                - "csv": CSV format with ",", "\\t", or " " as field separators and CRLF or LF
                  as line separator. Double quotes for escaping strings. Each line is a sensor
                  reading. Callback receives {"csv": "line_content"}.
                - "binary": Raw binary data. Callback receives {"binary": numpy_array}.
        """
        self.host = host
        self.port = port
        self.timeout = timeout
        self.data_format = data_format.lower()
        self.logger = logger
        self._is_started = False

        if self.data_format not in ["json", "csv", "binary"]:
            raise RemoteSensorConfigError(f"Invalid data_format: {data_format}. Must be 'json', 'csv', or 'binary'")

        # This callback doesn't require a lock as long as we're running on CPython
        self._datapoint_callback: Optional[Callable[[Dict[Any, Any]], None]] = None
        self._server = None
        self._loop = None
        self._server_thread = None
        self._stop_event = None
        self._client: Optional[websockets.ServerConnection] = None
        self._client_lock = None

    def start(self) -> None:
        """Start the WebSocket server."""
        if self._is_started:
            logger.warning("RemoteSensor is already started")
            return

        self._open_sensor()
        self._is_started = True
        logger.info(f"RemoteSensor started on {self.host}:{self.port}")

    def _open_sensor(self) -> None:
        """Start the WebSocket server."""
        self._server_thread = threading.Thread(target=self._start_server_thread, daemon=True)
        self._server_thread.start()

        # Wait for server to start
        start_time = time.time()
        start_timeout = 10
        while self._server is None and time.time() - start_time < start_timeout:
            if self._server is not None:
                break
            time.sleep(0.1)

        if self._server is None:
            raise RemoteSensorOpenError(f"Failed to start WebSocket server on {self.host}:{self.port}")

    def _start_server_thread(self) -> None:
        """Run WebSocket server in its own thread with event loop."""
        try:
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            self._loop.run_until_complete(self._start_server())
        except Exception as e:
            logger.error(f"WebSocket server thread error: {e}")
        finally:
            if self._loop and not self._loop.is_closed():
                self._loop.close()

    async def _start_server(self) -> None:
        """Start the WebSocket server."""
        try:
            self._stop_event = asyncio.Event()
            self._client_lock = asyncio.Lock()
            self._stop_event.clear()

            self._server = await websockets.serve(
                self._ws_handler,
                self.host,
                self.port,
                open_timeout=self.timeout,
                ping_timeout=self.timeout,
                close_timeout=self.timeout,
                ping_interval=20,
            )

            logger.info(f"RemoteSensor WebSocket server started on {self.host}:{self.port}")

            await self._stop_event.wait()

        except Exception as e:
            logger.error(f"Error starting WebSocket server: {e}")
            raise
        finally:
            if self._server:
                self._server.close()
                await self._server.wait_closed()

    async def _ws_handler(self, conn: websockets.ServerConnection) -> None:
        """Handle a connected WebSocket client. Only one client allowed at a time."""
        client_addr = f"{conn.remote_address[0]}:{conn.remote_address[1]}"

        async with self._client_lock:
            if self._client is not None:
                # Reject the new client
                logger.warning(f"Rejecting client {client_addr}: only one client allowed at a time")
                try:
                    await conn.send(json.dumps({"error": "Server busy", "message": "Only one client connection allowed at a time", "code": 1000}))
                    await conn.close(code=1000, reason="Server busy - only one client allowed")
                except Exception as e:
                    logger.warning(f"Error sending rejection message to {client_addr}: {e}")
                return

            # Accept the client
            self._client = conn

        logger.info(f"Client connected: {client_addr}")

        try:
            # Send welcome message
            try:
                await self._send_to_client({
                    "status": "connected",
                    "message": "You are now connected to the RemoteSensor server",
                    "data_format": self.data_format,
                })
            except Exception as e:
                logger.warning(f"Could not send welcome message to {client_addr}: {e}")

            async for message in conn:
                datapoints = await self._parse_message(message)
                if datapoints is not None and self._datapoint_callback is not None:
                    # Handle both single datapoint and list of datapoints
                    if not isinstance(datapoints, list):
                        datapoints = [datapoints]

                    for datapoint in datapoints:
                        try:
                            await self._loop.run_in_executor(None, self._datapoint_callback, datapoint)
                        except Exception as e:
                            logger.error(f"Error in datapoint callback: {e}")

        except websockets.exceptions.ConnectionClosed:
            logger.info(f"Client disconnected: {client_addr}")
        except Exception as e:
            logger.warning(f"Error handling client {client_addr}: {e}")
        finally:
            async with self._client_lock:
                if self._client == conn:
                    self._client = None
                    logger.info(f"Client removed: {client_addr}")

    async def _parse_message(self, message):
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
        if not self._is_started:
            logger.warning("RemoteSensor is not started")
            return

        self._close_sensor()
        self._is_started = False
        logger.info("RemoteSensor stopped")

    def _close_sensor(self) -> None:
        """Stop the WebSocket server."""
        if self._loop is not None and self._server is not None:
            try:
                future = asyncio.run_coroutine_threadsafe(self._stop_and_disconnect_client(), self._loop)
                future.result(timeout=1.0)
            except CancelledError:
                logger.debug(f"Error stopping WebSocket server: CancelledError")
            except TimeoutError:
                logger.debug(f"Error stopping WebSocket server: TimeoutError")
            except Exception as e:
                logger.warning(f"Error stopping WebSocket server: {e}")

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
                # Send goodbye message before closing
                await self._send_to_client({
                    "status": "disconnecting",
                    "message": "Server is shutting down. Connection will be closed.",
                })
                # Give a brief moment for the message to be sent
                await asyncio.sleep(0.1)
            except Exception as e:
                logger.warning(f"Error closing client in stop event: {e}")
            finally:
                await self._client.close()

        if self._stop_event:
            self._stop_event.set()

    async def _send_to_client(self, message: dict) -> None:
        """Send a message to the connected client."""
        if self._client is None:
            raise ConnectionError("No client connected to send message to")

        if isinstance(message, dict):
            message = json.dumps(message)

        try:
            await self._client.send(message)
        except Exception as e:
            logger.warning(f"Error sending message to client: {e}")

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
        self._datapoint_callback = callback

    def is_started(self) -> bool:
        """Check if the sensor is started and running."""
        return self._is_started

    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()
        return False
