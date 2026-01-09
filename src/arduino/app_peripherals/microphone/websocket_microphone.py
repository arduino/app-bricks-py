# SPDX-FileCopyrightText: Copyright (C) ARDUINO SRL (http://www.arduino.cc)
#
# SPDX-License-Identifier: MPL-2.0

import json
import base64
import os
import threading
import queue
import numpy as np
import websockets
import asyncio
from concurrent.futures import CancelledError, TimeoutError, Future

from arduino.app_internal.core.peripherals import BPPCodec
from arduino.app_utils import Logger

from .base_microphone import BaseMicrophone, FormatPlain, FormatPacked
from .errors import MicrophoneConfigError, MicrophoneOpenError

logger = Logger("WebSocketMicrophone")


class WebSocketMicrophone(BaseMicrophone):
    """
    WebSocket Microphone implementation that hosts a WebSocket server.

    This microphone exposes a WebSocket server that receives audio chunks from connected
    clients. Only one client can be connected at a time.

    Clients must encode the audio data in PCM format and serialize the content in the
    binary format supported by BPPCodec.

    Secure communication with the WebSocket server is supported in three security modes:
    - Security disabled (empty secret)
    - Authenticated (secret + encrypt=False) - HMAC-SHA256
    - Authenticated + Encrypted (secret + encrypt=True) - ChaCha20-Poly1305

    Also, clients are expected to respect the sample rate, channels, format, and chunk
    size specified during initialization.
    """

    from .microphone import Microphone

    def __init__(
        self,
        port: int = 8080,
        timeout: int = 3,
        certs_dir_path: str = "/app/certs",
        use_tls: bool = False,
        secret: str = "",
        encrypt: bool = False,
        sample_rate: int = Microphone.RATE_16K,
        channels: int = Microphone.CHANNELS_MONO,
        format: FormatPlain | FormatPacked = np.int16,
        buffer_size: int = Microphone.BUFFER_SIZE_BALANCED,
        auto_reconnect: bool = True,
    ):
        """
        Initialize WebSocket microphone server.

        Args:
            port (int): Port to bind the server to. Default: 8080.
            timeout (int): Connection timeout in seconds. Default: 3.
            certs_dir_path (str): Path to the directory containing TLS certificates.
                Default: "/app/certs".
            use_tls (bool): Enable TLS for secure connections. If True, 'encrypt' will
                be ignored. Use this for transport-level security with clients that can
                accept self-signed certificates or when supplying your own certificates.
                Default: False.
            secret (str): Secret key for authentication/encryption (empty = security disabled).
                Default: empty.
            encrypt (bool): Enable encryption (only effective if secret is provided).
                Default: False.
            sample_rate (int): Sample rate in Hz. Default: 16000.
            channels (int): Number of audio channels. Default: Microphone.CHANNELS_MONO - 1.
            format (FormatPlain | FormatPacked): Audio format as one of:
                - Type classes: np.int16, np.float32, np.uint8
                - dtype objects: np.dtype('<i2'), np.dtype('>f4')
                - Strings: 'int16', '<i2', '>f4', 'float32'
                - Tuple of (format, is_packed): to specify if the format is packed (e.g. 24-bit audio)
                Default: np.int16 - 16-bit signed platform-endian.
            buffer_size (int): Number of frames per buffer (default: 1024). This parameter is advisory,
                it's sent to clients to suggest an optimal buffer size but clients may ignore it.
                Default: Microphone.BUFFER_SIZE_BALANCED - 1024.
            auto_reconnect (bool): Enable automatic reconnection on failure.
        """
        super().__init__(sample_rate, channels, format, buffer_size, auto_reconnect)

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
            raise MicrophoneConfigError(f"Invalid port number: {port}")
        self.port = port
        if timeout <= 0:
            raise MicrophoneConfigError(f"Invalid timeout value: {timeout}")
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

        self._audio_queue = queue.Queue(10)
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

    def _open_microphone(self) -> None:
        """Start the WebSocket server."""
        server_future = Future()

        self._server_thread = threading.Thread(target=self._start_server_thread, args=(server_future,), daemon=True)
        self._server_thread.start()

        try:
            server_future.result(timeout=self.timeout)
            self.logger.info(f"WebSocket microphone server available on {self.url}, security: {self.security_mode}")
        except (PermissionError, Exception) as e:
            if self._server_thread.is_alive():
                self._server_thread.join(timeout=1.0)
            if isinstance(e, PermissionError):
                raise MicrophoneOpenError(f"Permission denied when attempting to bind WebSocket server on {self.url}")
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

        self._set_status("connected", {"client_address": client_addr})
        self.logger.debug(f"Client connected: {client_addr}")

        try:
            # Send welcome message
            try:
                welcome = {
                    "status": "connected",
                    "message": "You are now connected to the microphone server",
                    "security_mode": self.security_mode,
                    "sample_rate": self.sample_rate,
                    "channels": self.channels,
                    "buffer_size": self.buffer_size,
                }

                # Add universal format details if available
                format_details = _get_format_details(self.format, self.format_is_packed)
                if format_details:
                    welcome.update(format_details)

                await self._send_to_client(json.dumps(welcome))
            except Exception as e:
                self.logger.warning(f"Failed to send welcome message: {e}")

            # Handle incoming messages
            async for message in conn:
                audio_chunk = self._parse_message(message)
                if audio_chunk is None:
                    continue

                # Drop old chunks until there's room for the new one
                while True:
                    try:
                        self._audio_queue.put_nowait(audio_chunk)
                        break
                    except queue.Full:
                        try:
                            # Drop oldest chunk and try again
                            self._audio_queue.get_nowait()
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
                    self._set_status("disconnected", {"client_address": client_addr})
                    self.logger.debug(f"Client removed: {client_addr}")

    def _parse_message(self, message: str | bytes) -> np.ndarray | None:
        """Parse WebSocket message to extract audio chunk."""
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

        return np.frombuffer(decoded, dtype=self.format)

    def _close_microphone(self) -> None:
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
                self._audio_queue.get_nowait()
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

    def _read_audio(self) -> np.ndarray | None:
        """Read a single audio chunk from the queue."""
        try:
            return self._audio_queue.get(timeout=0.01)
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


def _get_format_details(format: np.dtype, is_packed: bool = False) -> dict | None:
    """Get detailed format information for clients by introspecting the numpy dtype."""
    bit_depth = format.itemsize * 8

    # Determine sample format from dtype kind
    if format.kind == "i":  # signed integer
        sample_format = "signed_integer"
    elif format.kind == "u":  # unsigned integer
        sample_format = "unsigned_integer"
    elif format.kind == "f":  # floating point
        sample_format = "float"
    else:
        sample_format = "unknown"

    # Determine byte order
    if format.byteorder == "<":
        byte_order = "little_endian"
    elif format.byteorder == ">":
        byte_order = "big_endian"
    elif format.byteorder == "|":
        # Not applicable (single byte types like int8, uint8)
        byte_order = "n/a"
    else:
        # Native byte order ('='), determine from system
        import sys

        byte_order = "little_endian" if sys.byteorder == "little" else "big_endian"

    return {
        "format_type": sample_format,
        "format_bit_depth": bit_depth,
        "format_is_packed": is_packed,
        "format_byte_order": byte_order,
    }
