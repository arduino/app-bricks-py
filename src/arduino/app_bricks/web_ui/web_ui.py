# SPDX-FileCopyrightText: Copyright (C) ARDUINO SRL (http://www.arduino.cc)
#
# SPDX-License-Identifier: MPL-2.0

import os
import asyncio
import threading
from contextlib import asynccontextmanager
from typing import Any
from collections.abc import Callable

import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi_socketio import SocketManager

from arduino.app_utils import brick, Logger

logger = Logger("WebUI")


@brick
class WebUI:
    """Module for deploying a web server that can host a web application and expose APIs to its clients.

    It uses FastAPI, Uvicorn, and Socket.IO to serve static files (e.g., HTML/CSS/JS), handle REST API endpoints,
    and support real-time communication between the client and the server.

    """

    def __init__(
        self,
        addr: str = "0.0.0.0",
        port: int = 7000,
        ui_path_prefix: str = "",
        api_path_prefix: str = "",
        assets_dir_path: str = "/app/assets",
        certs_dir_path: str = "/app/certs",
        use_tls: bool = False,
        use_ssl: bool | None = None,  # Deprecated alias for use_tls
    ):
        """Initialize the web server.

        Args:
            addr (str, optional): Server bind address. Defaults to "0.0.0.0" (all interfaces).
            port (int, optional): Server port number. Defaults to 7000.
            ui_path_prefix (str, optional): URL prefix for UI routes. Defaults to "" (root).
            api_path_prefix (str, optional): URL prefix for API routes. Defaults to "" (root).
            assets_dir_path (str, optional): Path to static assets directory. Defaults to "/app/assets".
            certs_dir_path (str, optional): Path to TLS certificates directory. Defaults to "/app/certs".
            use_tls (bool, optional): Enable TLS/HTTPS. Defaults to False.
            use_ssl (bool, optional): Deprecated. Use use_tls instead. Defaults to None.
        """
        # Handle deprecated use_ssl parameter
        if use_ssl is not None:
            logger.warning("'use_ssl' parameter is deprecated. Use 'use_tls' instead.")
            use_tls = use_ssl

        @asynccontextmanager
        async def lifespan(app):
            await self._on_startup()
            yield

        self.app = FastAPI(title=__name__, openapi_url=None, lifespan=lifespan)
        self.sio = SocketManager(app=self.app, mount_location="/socket.io", socketio_path="", max_http_buffer_size=10 * 1024 * 1024)

        self._addr = addr

        def pick_free_port():
            import socket

            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("127.0.0.1", 0))
                return s.getsockname()[1]

        self._port = port if port != 0 else pick_free_port()
        self._ui_path_prefix = ui_path_prefix
        self._api_path_prefix = api_path_prefix
        self._assets_dir_path = os.path.abspath(assets_dir_path)
        self._certs_dir_path = os.path.abspath(certs_dir_path)
        self._use_tls = use_tls
        self._protocol = "https" if self._use_tls else "http"
        self._server: uvicorn.Server | None = None
        self._server_loop: asyncio.AbstractEventLoop | None = None
        self._on_connect_cb: Callable[[str], None] | None = None
        self._on_disconnect_cb: Callable[[str], None] | None = None
        self._on_message_cbs = {}
        self._on_message_cbs_lock = threading.Lock()

    @property
    def local_url(self) -> str:
        """Get the locally addressable URL of the web server.

        Returns:
            str: The server's URL (including protocol, address, and port).
        """
        return f"{self._protocol}://localhost:{self._port}"

    @property
    def url(self) -> str:
        """Get the externally addressable URL of the web server.

        Returns:
            str: The server's URL (including protocol, address, and port).
        """
        return f"{self._protocol}://{os.getenv('HOST_IP') or self._addr}:{self._port}"

    def start(self):
        """Start the web server asynchronously.

        This sets up static file routing and WebSocket event handlers, configures TLS if enabled, and launches the server using Uvicorn.

        Raises:
            RuntimeError: If 'index.html' is missing in the static assets directory.
            RuntimeError: If TLS is enabled but certificates fail to generate.
            RuntimeWarning: If the server is already running.
        """
        # Setup static routes and SocketIO events
        if os.path.exists(self._assets_dir_path):
            # Only if the HTML directory exists we check for 'index.html'
            if not os.path.exists(os.path.join(self._assets_dir_path, "index.html")):
                raise RuntimeError(f"'index.html' is required but was not found in {self._assets_dir_path}.")
            self._init_static_routes()
        self._init_socketio()

        config = uvicorn.Config(self.app, host=self._addr, port=self._port, log_level="warning")
        if self._use_tls:
            from arduino.app_utils.tls_cert_manager import TLSCertificateManager

            try:
                cert_path, key_path = TLSCertificateManager.get_or_create_certificates(certs_dir=self._certs_dir_path, common_name=self._addr)
                config.ssl_certfile = cert_path
                config.ssl_keyfile = key_path
            except Exception as e:
                raise RuntimeError("Failed to configure TLS certificate. Please check the certs directory.") from e

        self._server = uvicorn.Server(config)

    def stop(self):
        """Stop the web server gracefully.

        Waits up to 5 seconds for current requests to finish before terminating.
        """
        logger.debug("Stopping server...")
        if self._server:
            self._server.should_exit = True  # Ask to stop the server

    def execute(self):
        logger.debug(f"Serving static web files from {self._assets_dir_path}")
        if self._use_tls:
            logger.debug(f"Using TLS certificates from {self._certs_dir_path}")

        logger.debug("Starting server...")

        startup_log = "The application interface is available here:\n"
        startup_log += f"  - Local URL:   {self.local_url}"
        if os.getenv("HOST_IP"):
            startup_log += f"\n  - Network URL: {self.url}"
        logger.info(startup_log)

        try:
            self._server.run()
        except Exception as e:
            logger.exception(f"Error running server: {e}")

    def expose_api(self, method: str, path: str, function: Callable):
        """Register a route with the specified HTTP method and path.

        The path will be prefixed with the api_path_prefix configured during initialization.

        Args:
            method (str): HTTP method to use (e.g., "GET", "POST").
            path (str): URL path for the API endpoint (without the prefix).
            function (Callable): Function to execute when the route is accessed.
        """
        self.app.add_api_route(self._api_path_prefix + path, function, methods=[method])

    def on_connect(self, callback: Callable[[str], None]):
        """Register a callback for WebSocket connection events.

        The callback should accept a single argument: the session ID (sid) of the connected client.

        Args:
            callback (Callable[[str], None]): Function to call when a client connects. Receives the session ID (sid) as its only argument.

        """
        self._on_connect_cb = callback

    def on_disconnect(self, callback: Callable[[str], None]):
        """Register a callback for WebSocket disconnection events.

        The callback should accept a single argument: the session ID (sid) of the disconnected client.

        Args:
            callback (Callable[[str], None]): Function to call when a client disconnects. Receives the session ID (sid) as its only argument.

        """
        self._on_disconnect_cb = callback

    def on_message(self, message_type: str, callback: Callable[[str, Any], Any]):
        """Register a callback function for a specific WebSocket message type received by clients.

        The client should send messages named as message_type for this callback to be triggered.

        If a response is returned by the callback, it will be sent back to the client
        with a message type suffix "_response".

        Args:
            message_type (str): The message type name to listen for.
            callback (Callable[[str, Any], Any]): Function to handle the message. Receives two arguments:
                the session ID (sid) and the incoming message data.

        """
        with self._on_message_cbs_lock:
            if message_type in self._on_message_cbs:
                logger.warning(f"Overwriting existing listener for message '{message_type}'")
            self._on_message_cbs[message_type] = callback
        logger.debug(f"Registered listener for message '{message_type}'")

    def send_message(self, message_type: str, message: dict | str, room: str | None = None):
        """Send a message to connected WebSocket clients.

        Args:
            message_type (str): The name of the message event to emit.
            message (dict | str): The message payload to send (dict or str).
            room (str): The target Socket.IO room (defaults to all clients).

        """
        if self._server_loop is None or not self._server_loop.is_running():
            logger.debug("Failed to send WebSocket message: asyncio loop is not available.")
            return

        try:
            coro = self.sio.emit(message_type, message, room=room)
            asyncio.run_coroutine_threadsafe(coro, self._server_loop)
        except Exception as e:
            logger.exception(f"Failed to send WebSocket message '{message_type}': {e}")

    async def _on_startup(self):
        """
        This function is called by uvicorn when the server starts up, it is necessary to capture the running
        asyncio event loop and reuse it later for emitting socket.io events as it requires an asyncio context.
        """
        self._server_loop = asyncio.get_running_loop()

    def _init_static_routes(self):
        from .cache import NonCachedStaticFiles

        url_path = self._ui_path_prefix.removesuffix("/") + "/"
        self.app.add_api_route(
            url_path,
            lambda: FileResponse(os.path.join(self._assets_dir_path, "index.html"), headers={"Cache-Control": "no-store"}),
            methods=["GET"],
            name="index",
        )
        self.app.mount(url_path, NonCachedStaticFiles(directory=self._assets_dir_path, html=True), name="static")

    def _init_socketio(self):
        @self.sio.on("connect")
        async def handle_connect(sid: str, environ: dict, auth: str):
            logger.debug(f"Client connected: {sid}")
            if self._on_connect_cb:
                try:
                    await asyncio.to_thread(self._on_connect_cb, sid)
                except Exception as e:
                    logger.exception(f"Error in 'on_connect' callback for {sid}: {e}")

        @self.sio.on("disconnect")
        async def handle_disconnect(sid: str, reason: str):
            logger.debug(f"Client disconnected ({reason}): {sid}")
            if self._on_disconnect_cb:
                try:
                    await asyncio.to_thread(self._on_disconnect_cb, sid)
                except Exception as e:
                    logger.exception(f"Error in 'on_disconnect' callback for {sid}: {e}")

        @self.sio.on("enter_room")
        async def handle_enter_room(sid: str, room: str):
            logger.debug(f"Client {sid} entering room {room}")
            await self.sio.enter_room(sid, room)

        @self.sio.on("leave_room")
        async def handle_leave_room(sid: str, room: str):
            logger.debug(f"Client {sid} leaving room {room}")
            await self.sio.leave_room(sid, room)

        @self.sio.on("*")
        async def handle_generic_event(event: str, sid: str, data: dict):
            """Handles generic messages from clients intended for the registered callbacks."""
            logger.debug(f"Received event'{event}' from {sid} containing: {data}")

            with self._on_message_cbs_lock:
                callback = self._on_message_cbs.get(event)

            if callback:

                async def run_callback_async():
                    try:
                        # Assuming the callback expects the payload as its argument
                        result = await asyncio.to_thread(callback, sid, data)
                        logger.debug(f"Successfully executed callback for '{event}'")
                        if result is not None:
                            logger.debug(f"Callback for '{event}' returned: {result}")
                            await self.sio.emit(f"{event}_response", result, room=sid)
                    except Exception as e:
                        logger.exception(f"Failed to execute callback for '{event}': {e}")
                        await self.sio.emit("error", f"Failed to execute callback for '{event}': {e}", room=sid)

                self.sio.start_background_task(run_callback_async)
            else:
                logger.warning(f"No listener registered for '{event}'")
                await self.sio.emit("error", f"No listener registered for '{event}'", room=sid)
