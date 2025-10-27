#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (C) 2025 ARDUINO SA <http://www.arduino.cc>
#
# SPDX-License-Identifier: MPL-2.0

"""
WebSocket Camera Proxy

This example demonstrates how to use a WebSocketCamera as a proxy/relay.
It receives frames from clients on one WebSocket server (127.0.0.1:8080) and
forwards them as raw JPEG binary data to a TCP server (127.0.0.1:5000) at 30fps.

Usage:
    python websocket_camera_proxy.py [--input-port PORT] [--output-host HOST] [--output-port PORT]
"""

import asyncio
import logging
import argparse
import signal
import sys
import time

import os

import numpy as np

from arduino.app_peripherals.camera import Camera
from arduino.app_utils.image.image_editor import ImageEditor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global variables for graceful shutdown
running = False
camera = None
output_writer = None
output_reader = None


def signal_handler(signum, frame):
    """Handle interrupt signals."""
    global running
    logger.info("Received signal, initiating shutdown...")
    running = False


async def connect_output_tcp(output_host: str, output_port: int):
    """Connect to the output TCP server."""
    global output_writer, output_reader
    
    logger.info(f"Connecting to output server at {output_host}:{output_port}...")
    
    try:
        output_reader, output_writer = await asyncio.open_connection(
            output_host, output_port
        )
        logger.info("Connected successfully to output server")
        
    except Exception as e:
        raise Exception(f"Failed to connect to output server: {e}")


async def forward_frame(frame: np.ndarray):
    """Forward a frame to the output TCP server as raw JPEG."""
    global output_writer
    
    if not output_writer or output_writer.is_closing():
        return
    
    try:
        output_writer.write(frame.tobytes())
        await output_writer.drain()
        
    except ConnectionResetError:
        logger.warning("Output connection reset while forwarding frame")
        output_writer = None
    except Exception as e:
        logger.error(f"Error forwarding frame: {e}")


async def camera_loop(fps: int):
    """Main camera capture and forwarding loop."""
    global running, camera
    
    frame_interval = 1.0 / fps
    last_frame_time = time.time()
    
    try:
        camera.start()
    except Exception as e:
        logger.error(f"Failed to start WebSocketCamera: {e}")
        return
    
    while running:
        try:
            # Read frame from WebSocketCamera
            frame = camera.capture()
            # frame = ImageEditor.compress_to_jpeg(frame, 80.1)
            if frame is None:
                # No frame available, small delay to avoid busy waiting
                await asyncio.sleep(0.01)
                continue
            
            # Rate limiting
            current_time = time.time()
            time_since_last = current_time - last_frame_time
            if time_since_last < frame_interval:
                await asyncio.sleep(frame_interval - time_since_last)
            
            last_frame_time = time.time()
            
            if output_writer is None or output_writer.is_closing():
                # Output connection is not available, give room to the other tasks
                await asyncio.sleep(0.01)
            else:
                # Forward frame if output connection is available
                await forward_frame(frame)
            
        except Exception as e:
            logger.error(f"Error in camera loop: {e}")
            await asyncio.sleep(1.0)


async def maintain_output_connection(output_host: str, output_port: int, reconnect_delay: float):
    """Maintain TCP connection to output server with automatic reconnection."""
    global running, output_writer, output_reader

    while running:
        try:
            await connect_output_tcp(output_host, output_port)
            
            # Keep monitoring
            while running and output_writer and not output_writer.is_closing():
                await asyncio.sleep(1.0)
                
            logger.info("Lost connection to output server")
                
        except Exception as e:
            logger.error(e)
        finally:
            # Clean up connection
            if output_writer:
                try:
                    output_writer.close()
                    await output_writer.wait_closed()
                except:
                    pass
                output_writer = None
                output_reader = None
        
        # Wait before reconnecting
        if running:
            logger.info(f"Reconnecting to output server in {reconnect_delay} seconds...")
            await asyncio.sleep(reconnect_delay)


async def main():
    """Main function."""
    global running, camera
    
    parser = argparse.ArgumentParser(description="WebSocket Camera Proxy")
    parser.add_argument("--input-host", default="localhost",
                       help="WebSocketCamera input host (default: localhost)")
    parser.add_argument("--input-port", type=int, default=8080,
                       help="WebSocketCamera input port (default: 8080)")
    parser.add_argument("--output-host", default="localhost",
                       help="Output TCP server host (default: localhost)")
    parser.add_argument("--output-port", type=int, default=5000,
                       help="Output TCP server port (default: 5000)")
    parser.add_argument("--fps", type=int, default=30,
                       help="Target FPS for forwarding (default: 30)")
    parser.add_argument("--quality", type=int, default=80,
                       help="JPEG quality 1-100 (default: 80)")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Setup global variables
    running = True
    reconnect_delay = 2.0
    
    logger.info(f"Starting WebSocket camera proxy")
    logger.info(f"Input: WebSocketCamera on port {args.input_port}")
    logger.info(f"Output: TCP server at {args.output_host}:{args.output_port}")
    logger.info(f"Target FPS: {args.fps}")
    
    from arduino.app_utils.image.image_editor import compressed_to_jpeg
    camera = Camera(f"ws://{args.input_host}:{args.input_port}", adjuster=compressed_to_jpeg(80))
    # camera = Camera(f"ws://{args.input_host}:{args.input_port}")
    
    try:
        # Start camera input and output connection tasks
        camera_task = asyncio.create_task(camera_loop(args.fps))
        connection_task = asyncio.create_task(maintain_output_connection(args.output_host, args.output_port, reconnect_delay))
        
        # Run both tasks concurrently
        await asyncio.gather(camera_task, connection_task)
        
    except KeyboardInterrupt:
        logger.info("Received interrupt signal, shutting down...")
    finally:
        running = False
        
        # Close output TCP connection
        if output_writer:
            try:
                output_writer.close()
                await output_writer.wait_closed()
            except Exception as e:
                logger.warning(f"Error closing TCP connection: {e}")
        
        # Close camera
        if camera:
            try:
                camera.stop()
                logger.info("Camera closed")
            except Exception as e:
                logger.warning(f"Error closing camera: {e}")
        
        logger.info("Camera proxy stopped")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)