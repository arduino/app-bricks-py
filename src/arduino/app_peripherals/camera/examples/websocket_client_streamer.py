# SPDX-FileCopyrightText: Copyright (C) 2025 ARDUINO SA <http://www.arduino.cc>
#
# SPDX-License-Identifier: MPL-2.0

import asyncio
import websockets
import base64
import json
import logging
import argparse
import signal
import sys
import time

from arduino.app_peripherals.camera import Camera
from arduino.app_utils.image.image_editor import ImageEditor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

FRAME_WIDTH = 640
FRAME_HEIGHT = 480


class WebCamStreamer:
    """
    WebSocket client that streams local webcam feed to a WebSocketCamera server.
    """
    
    def __init__(self, host: str = "localhost", port: int = 8080, 
                 camera_id: int = 0, fps: int = 30, quality: int = 80):
        """
        Initialize the webcam streamer.
        
        Args:
            host: WebSocket server host
            port: WebSocket server port
            camera_id: Local camera device ID (usually 0 for default camera)
            fps: Target frames per second for streaming
            quality: JPEG quality (1-100, higher = better quality)
        """
        self.host = host
        self.port = port
        self.camera_id = camera_id
        self.fps = fps
        self.quality = quality
        
        self.websocket_url = f"ws://{host}:{port}"
        self.frame_interval = 1.0 / fps
        self.reconnect_delay = 2.0
        
        self.running = False
        self.camera = None
        self.websocket = None
        self.server_frame_format = "base64"
        
    async def start(self):
        """Start the webcam streamer."""
        self.running = True
        logger.info(f"Starting webcam streamer (camera_id={self.camera_id}, fps={self.fps})")
        
        camera_task = asyncio.create_task(self._camera_loop())
        websocket_task = asyncio.create_task(self._websocket_loop())
        
        try:
            await asyncio.gather(camera_task, websocket_task)
        except KeyboardInterrupt:
            logger.info("Received interrupt signal, shutting down...")
        finally:
            await self.stop()
    
    async def stop(self):
        """Stop the webcam streamer."""
        logger.info("Stopping webcam streamer...")
        self.running = False
        
        if self.websocket:
            try:
                await self.websocket.close()
            except Exception as e:
                logger.warning(f"Error closing WebSocket: {e}")
        
        if self.camera:
            self.camera.stop()
            logger.info("Camera stopped")
        
        logger.info("Webcam streamer stopped")
    
    async def _camera_loop(self):
        """Main camera capture loop."""
        logger.info(f"Opening camera {self.camera_id}...")
        self.camera = Camera(self.camera_id, resolution=(FRAME_WIDTH, FRAME_HEIGHT), fps=self.fps)
        self.camera.start()
        
        if not self.camera.is_started():
            logger.error(f"Failed to open camera {self.camera_id}")
            return
        
        logger.info("Camera opened successfully")
        
        last_frame_time = time.time()
        
        while self.running:
            try:
                frame = self.camera.capture()
                if frame is None:
                    logger.warning("Failed to capture frame")
                    await asyncio.sleep(0.1)
                    continue

                # Rate limiting to enforce frame rate
                current_time = time.time()
                time_since_last = current_time - last_frame_time
                if time_since_last < self.frame_interval:
                    await asyncio.sleep(self.frame_interval - time_since_last)
                
                last_frame_time = time.time()
                
                if self.websocket:
                    try:
                        await self._send_frame(frame)
                    except websockets.exceptions.ConnectionClosed:
                        logger.warning("WebSocket connection lost during frame send")
                        self.websocket = None
                
                await asyncio.sleep(0.001)
                
            except Exception as e:
                logger.error(f"Error in camera loop: {e}")
                await asyncio.sleep(1.0)
    
    async def _websocket_loop(self):
        """Main WebSocket connection loop with automatic reconnection."""
        while self.running:
            try:
                await self._connect_websocket()
                await self._handle_websocket_messages()
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
            finally:
                if self.websocket:
                    try:
                        await self.websocket.close()
                    except:
                        pass
                    self.websocket = None
            
            if self.running:
                logger.info(f"Reconnecting in {self.reconnect_delay} seconds...")
                await asyncio.sleep(self.reconnect_delay)
    
    async def _connect_websocket(self):
        """Connect to the WebSocket server."""
        logger.info(f"Connecting to {self.websocket_url}...")
        
        try:
            self.websocket = await websockets.connect(
                self.websocket_url,
                ping_interval=20,
                ping_timeout=10,
                close_timeout=5
            )
            logger.info("WebSocket connected successfully")

        except Exception as e:
            raise
    
    async def _handle_websocket_messages(self):
        """Handle incoming WebSocket messages."""
        try:
            async for message in self.websocket:
                try:
                    data = json.loads(message)

                    if data.get("status") == "connected":
                        logger.info(f"Server welcome: {data.get('message', 'Connected')}")
                        self.server_frame_format = data.get('frame_format', 'base64')
                        logger.info(f"Server format: {self.server_frame_format}")
                    
                    elif data.get("status") == "disconnecting":
                        logger.info(f"Server goodbye: {data.get('message', 'Disconnecting')}")
                        break
                    
                    elif data.get("error"):
                        logger.warning(f"Server error: {data.get('message', 'Unknown error')}")
                        if data.get("code") == 1000:  # Server busy
                            break
                    
                    else:
                        logger.warning(f"Received unknown message: {data}")

                except json.JSONDecodeError:
                    logger.warning(f"Received non-JSON message: {message[:100]}")

        except websockets.exceptions.ConnectionClosed:
            logger.info("WebSocket connection closed by server")
        except Exception as e:
            logger.error(f"Error handling WebSocket messages: {e}")
            raise
    
    async def _send_frame(self, frame):
        """Send a frame to the WebSocket server using the server's preferred format."""
        try:
            if self.server_frame_format == "binary":
                # Encode frame as JPEG and send binary data
                encoded_frame = ImageEditor.compress_to_jpeg(frame)
                await self.websocket.send(encoded_frame.tobytes())
                
            elif self.server_frame_format == "base64":
                # Encode frame as JPEG and send base64 data
                encoded_frame = ImageEditor.compress_to_jpeg(frame)
                frame_b64 = base64.b64encode(encoded_frame.tobytes()).decode('utf-8')
                await self.websocket.send(frame_b64)
            
            elif self.server_frame_format == "json":
                # Encode frame as JPEG, base64 encode and wrap in JSON
                encoded_frame = ImageEditor.compress_to_jpeg(frame)
                frame_b64 = base64.b64encode(encoded_frame.tobytes()).decode('utf-8')
                message = json.dumps({"image": frame_b64})
                await self.websocket.send(message)
                
            else:
                logger.warning(f"Unknown server frame format: {self.server_frame_format}")
            
        except websockets.exceptions.ConnectionClosed:
            logger.warning("WebSocket connection closed while sending frame")
            raise
        except Exception as e:
            logger.error(f"Error sending frame: {e}")


def signal_handler(signum, frame):
    """Handle interrupt signals."""
    logger.info("Received signal, initiating shutdown...")
    sys.exit(0)


async def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="WebSocket Camera Client Streamer")
    parser.add_argument("--host", default="localhost", help="WebSocket server host (default: localhost)")
    parser.add_argument("--port", type=int, default=8080, help="WebSocket server port (default: 8080)")
    parser.add_argument("--camera", type=int, default=0, help="Camera device ID (default: 0)")
    parser.add_argument("--fps", type=int, default=30, help="Target FPS (default: 30)")
    parser.add_argument("--quality", type=int, default=80, help="JPEG quality 1-100 (default: 80)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Create and start streamer
    streamer = WebCamStreamer(
        host=args.host,
        port=args.port,
        camera_id=args.camera,
        fps=args.fps,
        quality=args.quality
    )
    
    try:
        await streamer.start()
    except KeyboardInterrupt:
        pass
    finally:
        await streamer.stop()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)
