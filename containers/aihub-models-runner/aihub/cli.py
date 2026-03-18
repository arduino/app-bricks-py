# SPDX-FileCopyrightText: Copyright (C) ARDUINO SRL (http://www.arduino.cc)
#
# SPDX-License-Identifier: MPL-2.0

import argparse
from typing import Any, Dict, List, Optional


def create_parser() -> argparse.ArgumentParser:
    """Create and return the argument parser."""
    parser = argparse.ArgumentParser(
        description="AIHub Application",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default: GStreamer input with MJPEG output
  python main.py

  # WebSocket-only mode
  python main.py --input websocket --output websocket

  # GStreamer input with both MJPEG and WebSocket outputs
  python main.py --output mjpeg websocket --mjpeg-port 8080 --ws-output-port 9000

  # Custom GStreamer source
  python main.py --gst-source "v4l2src device=/dev/video2"
""",
    )

    # Input Configuration
    input_group = parser.add_argument_group("Input Configuration")
    input_group.add_argument(
        "--input",
        "-i",
        type=str,
        choices=["gstreamer", "websocket"],
        default="gstreamer",
        dest="input_type",
        help="Input source type (default: gstreamer)",
    )

    # Output Configuration
    output_group = parser.add_argument_group("Output Configuration")
    output_group.add_argument(
        "--output",
        "-o",
        type=str,
        nargs="+",
        choices=["mjpeg", "websocket"],
        default=["mjpeg"],
        dest="output_types",
        help="Output sink type(s), can specify multiple (default: mjpeg)",
    )

    # GStreamer Input Options
    gst_group = parser.add_argument_group("GStreamer Input Options")
    gst_group.add_argument(
        "--gst-source",
        type=str,
        default="v4l2src device=/dev/video0",
        help='GStreamer source element (default: "v4l2src device=/dev/video0"). '
        'Examples: "qtiqmmfsrc name=camsrc camera=0", "v4l2src device=/dev/video2"',
    )
    gst_group.add_argument(
        "--gst-width",
        type=int,
        default=1024,
        help="Video width (default: 1024)",
    )
    gst_group.add_argument(
        "--gst-height",
        type=int,
        default=768,
        help="Video height (default: 768)",
    )

    # WebSocket Input Options
    ws_input_group = parser.add_argument_group("WebSocket Input Options")
    ws_input_group.add_argument(
        "--ws-input-host",
        type=str,
        default="0.0.0.0",
        help="Host to bind WebSocket input server (default: 0.0.0.0)",
    )
    ws_input_group.add_argument(
        "--ws-input-port",
        type=int,
        default=5000,
        help="Port for WebSocket input server (default: 5000)",
    )

    # WebSocket Output Options
    ws_output_group = parser.add_argument_group("WebSocket Output Options")
    ws_output_group.add_argument(
        "--ws-output-host",
        type=str,
        default="0.0.0.0",
        help="Host to bind WebSocket output server (default: 0.0.0.0)",
    )
    ws_output_group.add_argument(
        "--ws-output-port",
        type=int,
        default=5001,
        help="Port for WebSocket output server (default: 5001)",
    )

    # MJPEG Output Options
    mjpeg_group = parser.add_argument_group("MJPEG Output Options")
    mjpeg_group.add_argument(
        "--mjpeg-host",
        type=str,
        default="0.0.0.0",
        help="Host to bind MJPEG server (default: 0.0.0.0)",
    )
    mjpeg_group.add_argument(
        "--mjpeg-port",
        type=int,
        default=5002,
        help="Port for MJPEG server (default: 5002)",
    )

    # Logging Options
    logging_group = parser.add_argument_group("Logging Options")
    logging_group.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output (FPS tracking, connection events)",
    )

    return parser


def parse_args(args: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Parse command line arguments and return as dictionary suitable for AIHubApp.

    Args:
        args: Optional list of arguments (for testing). If None, uses sys.argv.

    Returns:
        Dictionary with keys matching AIHubApp constructor kwargs.
    """
    parser = create_parser()
    parsed = parser.parse_args(args)

    # Convert to dict with proper key prefixes for AIHubApp
    result = {
        "input_type": parsed.input_type,
        "output_types": parsed.output_types,
        # GStreamer input options (gst_ prefix)
        "gst_source": parsed.gst_source,
        "gst_width": parsed.gst_width,
        "gst_height": parsed.gst_height,
        # WebSocket input options (ws_input_ prefix)
        "ws_input_host": parsed.ws_input_host,
        "ws_input_port": parsed.ws_input_port,
        # MJPEG output options (mjpeg_ prefix)
        "mjpeg_host": parsed.mjpeg_host,
        "mjpeg_port": parsed.mjpeg_port,
        # WebSocket output options (ws_output_ prefix)
        "ws_output_host": parsed.ws_output_host,
        "ws_output_port": parsed.ws_output_port,
        # Logging options
        "verbose": parsed.verbose,
    }

    return result
