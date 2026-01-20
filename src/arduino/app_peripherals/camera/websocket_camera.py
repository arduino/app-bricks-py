# SPDX-FileCopyrightText: Copyright (C) ARDUINO SRL (http://www.arduino.cc)
#
# SPDX-License-Identifier: MPL-2.0

import json
import base64
import os
import threading
import queue
import numpy as np
import cv2
import websockets
import asyncio
from urllib.parse import urlparse, parse_qs
from collections.abc import Callable
from concurrent.futures import CancelledError, TimeoutError, Future

from arduino.app_internal.core.peripherals import BPPCodec
from arduino.app_utils import Logger

from .base_camera import BaseCamera
from .errors import CameraConfigError, CameraOpenError

logger = Logger("WebSocketCamera")


class WebSocketCamera(BaseCamera):
    """
    WebSocket Camera implementation that hosts a WebSocket server.

    This camera acts as a WebSocket server that receives frames from connected clients.
    Only one client can be connected at a time.

    Clients must encode video frames in one of these formats:
    - JPEG
    - PNG
    - WebP
    - BMP
    - TIFF
    The video frames must then be serialized in the binary format supported by BPPCodec.

    Secure communication with the WebSocket server is supported in three security modes:
    - Security disabled (empty secret)
    - Authenticated (secret + encrypt=False) - HMAC-SHA256
    - Authenticated + Encrypted (secret + encrypt=True) - ChaCha20-Poly1305

    When connecting, clients can specify a "client_name" parameter in the URL query string
    to identify themselves. This name will be sanitized to allow only alphanumeric chars,
    whitespace, hyphens, and underscores, and limit its length to 64 characters.
    """

    def __init__(
        self,
        port: int = 8080,
        timeout: int = 3,
        certs_dir_path: str = "/app/certs",
        use_tls: bool = False,
        secret: str = "",
        encrypt: bool = False,
        resolution: tuple[int, int] = (640, 480),
        fps: int = 10,
        adjustments: Callable[[np.ndarray], np.ndarray] | None = None,
        auto_reconnect: bool = True,
    ):
        """
        Initialize WebSocket camera server with security options.

        Args:
            port (int): Port to bind the server to
            timeout (int): Connection timeout in seconds
            certs_dir_path (str): Path to the directory containing TLS certificates
            use_tls (bool): Enable TLS for secure connections. If True, 'encrypt' will
                be ignored. Use this for transport-level security with clients that can
                accept self-signed certificates or when supplying your own certificates.
            secret (str): Secret key for authentication/encryption (empty = security disabled)
            encrypt (bool): Enable encryption (only effective if secret is provided)
            resolution (tuple[int, int]): Resolution as (width, height)
            fps (int): Frames per second to capture
            adjustments (Callable[[np.ndarray], np.ndarray] | None): Function to adjust frames
            auto_reconnect (bool): Enable automatic reconnection on failure
        """
        super().__init__(resolution, fps, adjustments, auto_reconnect)

        if use_tls and encrypt:
            logger.warning("Encryption is redundant over TLS connections, disabling encryption.")
            encrypt = False

        self.codec = BPPCodec(secret, encrypt)
        self.secret = secret
        self.encrypt = encrypt
        self.logger = logger
        self.name = self.__class__.__name__

        # Address and port configuration
        self.use_tls = use_tls
        self.protocol = "wss" if use_tls else "ws"
        self._bind_ip = "0.0.0.0"
        host_ip = os.getenv("HOST_IP")
        self.ip = host_ip if host_ip is not None else self._bind_ip
        if port < 0 or port > 65535:
            raise CameraConfigError(f"Invalid port number: {port}")
        self.port = port
        if timeout <= 0:
            raise CameraConfigError(f"Invalid timeout value: {timeout}")
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

        self._frame_queue = queue.Queue(1)
        self._server = None
        self._loop = None
        self._server_thread = None
        self._stop_event = asyncio.Event()
        self._client: websockets.ServerConnection | None = None
        self._client_lock = asyncio.Lock()

    @property
    def url(self) -> str:
        """Return the WebSocket server address."""
        return f"{self.protocol}://{self.ip}:{self.port}"

    @property
    def security_mode(self) -> str:
        """Return current security mode for logging/debugging."""
        if not self.secret:
            return "none"
        elif self.encrypt:
            return "encrypted (ChaCha20-Poly1305)"
        else:
            return "authenticated (HMAC-SHA256)"

    def _open_camera(self) -> None:
        """Start the WebSocket server."""
        server_future = Future()

        self._server_thread = threading.Thread(target=self._start_server_thread, args=(server_future,), daemon=True)
        self._server_thread.start()

        try:
            server_future.result(timeout=self.timeout)
            self.logger.info(f"WebSocket camera server available on {self.url}, security: {self.security_mode}")
        except (PermissionError, Exception) as e:
            if self._server_thread.is_alive():
                self._server_thread.join(timeout=1.0)
            if isinstance(e, PermissionError):
                raise CameraOpenError(f"Permission denied when attempting to bind WebSocket server on {self.url}")
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
                    max_size=5 * 1024 * 1024,  # Limit max message size for security
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
        # Extract and sanitize client_name from URL parameters
        client_name = "Unknown"
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
            except Exception as e:
                self.logger.debug(f"Failed to extract client_name from URL parameters: {e}")
            finally:
                self.name = client_name

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
                    "message": "Connected to camera server",
                    "security_mode": self.security_mode,
                    "resolution": self.resolution,
                    "fps": self.fps,
                }
                await self._send_to_client(json.dumps(welcome))
            except Exception as e:
                self.logger.warning(f"Failed to send welcome message: {e}")

            # Handle incoming messages
            async for message in conn:
                frame = self._parse_message(message)
                if frame is None:
                    continue

                # Drop old frames until there's room for the new one
                while True:
                    try:
                        self._frame_queue.put_nowait(frame)
                        break
                    except queue.Full:
                        try:
                            # Drop oldest frame and try again
                            self._frame_queue.get_nowait()
                        except queue.Empty:
                            continue

        except websockets.exceptions.ConnectionClosed:
            self.logger.debug(f"Client disconnected: {client_addr}")
        except Exception as e:
            self.logger.warning(f"Error handling client {client_addr}: {e}")
        finally:
            async with self._client_lock:
                if self._client == conn:
                    self._client = None
                    self._set_status("disconnected", {"client_address": client_addr, "client_name": client_name})
                    self.logger.debug(f"Client removed: {client_addr}")

    def _parse_message(self, message: websockets.Data) -> np.ndarray | None:
        """Parse WebSocket message to extract a video frame."""
        if isinstance(message, str):
            try:
                message = base64.b64decode(message)
            except Exception as e:
                self.logger.warning(f"Failed to decode string message using base64: {e}")
                return None

        decoded = self.codec.decode(message)
        if decoded is None:
            self.logger.warning("Failed to decode message")
            return None

        nparr = np.frombuffer(decoded, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
        return frame

    def _close_camera(self):
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

        # Clear frame queue
        try:
            while True:
                self._frame_queue.get_nowait()
        except queue.Empty:
            pass

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

    def _read_frame(self) -> np.ndarray | None:
        """Read a single frame from the queue."""
        try:
            return self._frame_queue.get(timeout=0.1)
        except queue.Empty:
            return None

    async def _send_to_client(self, message: bytes | str, client: websockets.ServerConnection | None = None):
        """Send a message to the connected client."""
        if isinstance(message, str):
            message = message.encode()

        encoded = self.codec.encode(message)

        # Keep a ref to current client to avoid locking
        client = client or self._client
        if client is None:
            raise ConnectionError("No client connected")

        try:
            await client.send(encoded)
        except websockets.ConnectionClosedOK:
            self.logger.warning("Client has already closed the connection")
        except websockets.ConnectionClosedError as e:
            self.logger.warning(f"Client has already closed the connection with error: {e}")
        except Exception:
            raise
