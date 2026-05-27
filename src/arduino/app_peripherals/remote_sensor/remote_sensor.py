# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

import json
import base64
import os
import threading
import time
import websockets
import asyncio
from urllib.parse import urlparse, parse_qs
from typing import Callable, Literal
from concurrent.futures import CancelledError, ThreadPoolExecutor, Future

from arduino.app_internal.core.peripherals.bpp_codec import BPPCodec
from arduino.app_utils import Logger

from .errors import RemoteSensorOpenError, RemoteSensorConfigError

logger = Logger("RemoteSensor")


class RemoteSensor:
    """
    RemoteSensor implementation that hosts a WebSocket server.

    This sensor acts as a WebSocket server that receives telemetry data from a
    connected client. Only one client can be connected at a time.

    Communication uses the BPP (Binary Peripheral Protocol) in three security modes:
    - Security disabled (secret=None) - BPP with no authentication
    - Authenticated (secret + encrypt=False) - BPP with HMAC-SHA256
    - Authenticated + Encrypted (secret + encrypt=True) - BPP with ChaCha20-Poly1305

    By default, all modes use BPP framing. When security is disabled (secret=None),
    clients can opt out of BPP by connecting with the "raw=true" URL query parameter,
    allowing them to send raw bytes directly without BPP wrapping. This parameter
    is silently ignored when security is enabled.

    When connecting, clients can specify a "client_name" parameter in the URL query string
    to identify themselves. This name will be sanitized to allow only alphanumeric chars,
    whitespace, hyphens, and underscores, and limit its length to 64 characters.

    Each message is handed to the registered callback via the on_datapoint method.
    """

    def __init__(
        self,
        port: int = 8090,
        timeout: int = 3,
        certs_dir_path: str = "/app/certs",
        use_tls: bool = False,
        secret: str | None = None,
        encrypt: bool = False,
        auto_reconnect: bool = True,
    ):
        """
        Initialize RemoteSensor WebSocket server.

        Args:
            port (int): Port to bind the server to. Default: 8090.
            timeout (int): Connection timeout in seconds
            certs_dir_path (str): Path to the directory containing TLS certificates
            use_tls (bool): Enable TLS for secure connections. If True, 'encrypt' will
                be ignored. Use this for transport-level security with clients that can
                accept self-signed certificates or when supplying your own certificates.
            secret (str | None): Pre-shared secret key used for HMAC-SHA256
                authentication, or to derive the ChaCha20-Poly1305 key when
                encrypt is True. None disables security. Default: None.
            encrypt (bool): Enable ChaCha20-Poly1305 encryption. Requires a
                non-None secret; raises RuntimeError otherwise. Default: False.
            auto_reconnect (bool): Enable automatic reconnection on failure
        """
        if encrypt and secret is None:
            raise RuntimeError("Encryption requires a secret key.")

        if use_tls and encrypt:
            logger.warning("Encryption is redundant over TLS connections, disabling encryption.")
            encrypt = False

        self.codec = BPPCodec(secret or "", encrypt)
        self.secret = secret
        self.encrypt = encrypt
        self._client_raw = False
        self.logger = logger
        self.name = self.__class__.__name__

        # Auto-reconnection parameters
        self.auto_reconnect = auto_reconnect
        self.auto_reconnect_delay = 1.0
        self.first_connection_max_retries = 10

        # Address and port configuration
        self.use_tls = use_tls
        self.protocol = "wss" if use_tls else "ws"
        self._bind_ip = "0.0.0.0"
        host_ip = os.getenv("HOST_IP")
        self.ip = host_ip if host_ip is not None else self._bind_ip
        if port < 0 or port > 65535:
            raise RemoteSensorConfigError(f"Invalid port number: {port}")
        self.port = port
        if timeout <= 0:
            raise RemoteSensorConfigError(f"Invalid timeout value: {timeout}")
        self.timeout = timeout

        # TLS configuration
        if self.use_tls:
            import ssl
            from arduino.app_utils.tls_cert_manager import TLSCertificateManager

            try:
                cert_path, key_path = TLSCertificateManager.get_or_create_certificates(certs_dir=certs_dir_path)
                self._ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
                self._ssl_context.load_cert_chain(cert_path, key_path)
                logger.info(f"SSL context created with certificate: {cert_path}")
            except Exception as e:
                raise RuntimeError("Failed to configure TLS certificate. Please check certificates and the certs directory.") from e

        self._status: Literal["disconnected", "connected", "streaming", "paused"] = "disconnected"
        self._is_started = False
        self._sensor_lock = threading.Lock()
        self._server = None
        self._loop = None
        self._server_thread = None
        self._client: websockets.ServerConnection | None = None
        self._client_lock = asyncio.Lock()

        # Event handling
        # These callbacks don't require locking as long as we're running on CPython
        self._on_datapoint_cb: Callable[[bytes], None] | None = None
        self._on_status_changed_cb: Callable[[str, dict], None] | None = None
        self._event_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="RemoteSensorCallbackRunner")

    @property
    def status(self) -> Literal["disconnected", "connected", "streaming", "paused"]:
        """Read-only property for camera status."""
        return self._status

    @property
    def url(self) -> str:
        """Return the WebSocket server address."""
        return f"{self.protocol}://{self.ip}:{self.port}"

    @property
    def security_mode(self) -> str:
        """Return current security mode for logging/debugging."""
        if self.secret is None:
            return "none"
        elif self.encrypt:
            return "encrypted (ChaCha20-Poly1305)"
        else:
            return "authenticated (HMAC-SHA256)"

    def start(self) -> None:
        """Start the WebSocket server."""
        with self._sensor_lock:
            self.logger.info("Starting remote sensor...")

            attempt = 0
            while not self.is_started():
                try:
                    self._open_sensor()
                    self._is_started = True
                    self.logger.info(f"Successfully started {self.name}")
                except RemoteSensorOpenError as e:  # We consider this a fatal error so we don't retry
                    self.logger.error(f"Fatal error while starting {self.name}: {e}")
                    raise
                except Exception as e:
                    if not self.auto_reconnect:
                        raise
                    attempt += 1
                    if attempt >= self.first_connection_max_retries:
                        raise RemoteSensorOpenError(
                            f"Failed to start remote sensor {self.name} after {self.first_connection_max_retries} attempts, last error is: {e}"
                        )

                    delay = min(self.auto_reconnect_delay * (2 ** (attempt - 1)), 60)  # Exponential backoff
                    self.logger.warning(
                        f"Failed attempt {attempt}/{self.first_connection_max_retries} at starting remote sensor {self.name}: {e}. "
                        f"Retrying in {delay:.1f}s..."
                    )
                    time.sleep(delay)

    def on_datapoint(self, callback: Callable[[bytes], None]) -> None:
        """
        Register a callback function to be called when a datapoint is received.

        The callback function will be called with a single argument: the binary
        data received.

        Args:
            callback (Callable[[bytes], None]): A function that takes binary data
                and returns None.
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

    def stop(self) -> None:
        """Stop the WebSocket server."""
        with self._sensor_lock:
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

    def _open_sensor(self) -> None:
        """Start the WebSocket server."""
        server_future = Future()

        self._server_thread = threading.Thread(target=self._start_server_thread, args=(server_future,), daemon=True)
        self._server_thread.start()

        try:
            server_future.result(timeout=self.timeout)
            self.logger.info(f"WebSocket remote sensor server available on {self.url}, security: {self.security_mode}")
        except Exception as e:
            if self._server_thread.is_alive():
                self._server_thread.join(timeout=1.0)
            if isinstance(e, OSError):
                raise RemoteSensorOpenError(f"Failed to bind WebSocket server on {self.url}: {e}") from e
            raise

    def _start_server_thread(self, future: Future) -> None:
        """Run WebSocket server in its own thread with event loop."""
        try:
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            self._loop.run_until_complete(self._start_server(future))
        finally:
            if self._loop and not self._loop.is_closed():
                self._loop.close()
                self._loop = None

    async def _start_server(self, future: Future) -> None:
        """Start the WebSocket server."""
        try:
            self._server = await asyncio.wait_for(
                websockets.serve(
                    self._ws_handler,
                    self._bind_ip,
                    self.port,
                    open_timeout=self.timeout,
                    ping_timeout=self.timeout,
                    close_timeout=self.timeout,
                    ping_interval=20,
                    max_size=1 * 1024 * 1024,  # Limit max message size for security
                    ssl=self._ssl_context if self.use_tls else None,
                ),
                timeout=self.timeout,
            )

            # Get the actual port if OS assigned one (i.e. when port=0)
            if self.port == 0:
                server_socket = list(self._server.sockets)[0]
                self.port = server_socket.getsockname()[1]

            future.set_result(True)

            await self._server.wait_closed()

        except Exception as e:
            future.set_exception(e)
        finally:
            self._server = None

    async def _ws_handler(self, conn: websockets.ServerConnection) -> None:
        """Handle a connected WebSocket client. Only one client allowed at a time."""
        # Extract URL parameters: client_name and raw mode opt-in
        client_name = "Unknown"
        client_raw = False
        if conn.request:
            try:
                parsed_path = urlparse(conn.request.path)
                query_params = parse_qs(parsed_path.query)
                if "client_name" in query_params:
                    raw_name = query_params["client_name"][0]
                    # Sanitize: only allow alphanumeric, spaces, hyphens, underscores, and limit length
                    sanitized = "".join(c for c in raw_name if c.isalnum() or c in " -_")[:64]
                    if sanitized:
                        client_name = sanitized
                # Allow raw (no BPP) mode only when security is disabled
                if "raw" in query_params and (not query_params["raw"] or query_params["raw"][0].lower() != "false"):
                    if self.secret is None:
                        client_raw = True
                    else:
                        self.logger.warning("Client requested raw mode but security is enabled, ignoring.")
            except Exception as e:
                self.logger.debug(f"Failed to extract URL parameters: {e}")
            finally:
                self.name = client_name
                self._client_raw = client_raw

        client_addr = f"{conn.remote_address[0]}:{conn.remote_address[1]}"

        async with self._client_lock:
            if self._client is not None:
                # Reject the new client
                self.logger.warning(f"Rejecting client {client_addr}: only one client allowed at a time")
                try:
                    rejection = json.dumps({"error": "Server busy", "message": "Only one client connection allowed at a time", "code": 1000})
                    await self._send_to_client(rejection, client=conn)
                    await conn.close(code=1000, reason="Server busy, only one client allowed")
                except Exception as e:
                    self.logger.warning(f"Failed to send rejection message to {client_addr}: {e}")
                return

            # Accept the client
            self._client = conn

        self._set_status("connected", {"client_address": client_addr, "client_name": client_name})
        self.logger.debug(f"Client connected: {client_addr}")

        try:
            # Send welcome message
            try:
                welcome = {
                    "status": "connected",
                    "message": "You are now connected to the remote sensor server",
                    "security_mode": self.security_mode,
                }
                await self._send_to_client(json.dumps(welcome))
            except Exception as e:
                self.logger.warning(f"Failed to send welcome message: {e}")

            # Handle incoming messages
            async for message in conn:
                datapoint = self._parse_message(message)
                if datapoint is None or self._on_datapoint_cb is None or self._loop is None:
                    continue

                try:
                    await self._loop.run_in_executor(None, self._on_datapoint_cb, datapoint)
                except Exception as e:
                    self.logger.error(f"Error in datapoint callback: {e}")

        except websockets.exceptions.ConnectionClosed:
            self.logger.debug(f"Client disconnected: {client_addr}")
        except Exception as e:
            self.logger.warning(f"Error handling client {client_addr}: {e}")
        finally:
            async with self._client_lock:
                if self._client == conn:
                    self._client = None
                    self._client_raw = False
                    self._set_status("disconnected", {"client_address": client_addr, "client_name": client_name})
                    self.logger.debug(f"Client removed: {client_addr}")

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

    def _parse_message(self, message: websockets.Data) -> bytes | None:
        """Parse WebSocket message to extract a datapoint."""
        if isinstance(message, str):
            try:
                message = base64.b64decode(message)
            except Exception as e:
                self.logger.warning(f"Failed to decode string message using base64: {e}")
                return None

        if not self._client_raw:
            decoded = self.codec.decode(message)
            if decoded is None:
                self.logger.warning("Failed to decode message")
                return None
            message = decoded

        return message

    def _close_sensor(self) -> None:
        """Stop the WebSocket server."""
        if self._loop and not self._loop.is_closed() and self._loop.is_running():
            try:
                future = asyncio.run_coroutine_threadsafe(self._disconnect_and_stop(), self._loop)
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

    async def _disconnect_and_stop(self):
        """Cleanly disconnect client with goodbye message and stop the server."""
        async with self._client_lock:
            if self._client:
                try:
                    self.logger.debug("Disconnecting client...")
                    goodbye = json.dumps({"status": "disconnecting", "message": "Server is shutting down"})
                    await self._send_to_client(goodbye)
                except Exception as e:
                    self.logger.warning(f"Failed to send goodbye message: {e}")
                finally:
                    if self._client:
                        await self._client.close()
                        self.logger.debug("Client connection closed")

        if self._server:
            self._server.close()

    async def _send_to_client(self, message: bytes | str, client: websockets.ServerConnection | None = None):
        """Send a message to the connected client."""
        if isinstance(message, str):
            message = message.encode()

        data = message if self._client_raw else self.codec.encode(message)

        # Keep a ref to current client to avoid locking
        client = client or self._client
        if client is None:
            raise ConnectionError("No client connected")

        try:
            await client.send(data)
        except websockets.ConnectionClosedOK:
            self.logger.warning("Client has already closed the connection")
        except websockets.ConnectionClosedError as e:
            self.logger.warning(f"Client has already closed the connection with error: {e}")
        except Exception:
            raise

    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()
        return False
