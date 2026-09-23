# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""Base classes of the bricks running Edge Impulse models on the `arduino:edge_impulse` inference service.

`EdgeImpulseModel` holds the model and its connection for any input, `VideoInference` runs it on the frames of a
camera and serves the video with an overlay; the video bricks of the library build on them, and so can yours.
"""

from .model import EdgeImpulseModel as EdgeImpulseModel
from .video import AllDetectionsCallback as AllDetectionsCallback
from .video import DetectionCallback as DetectionCallback
from .video import VideoInference as VideoInference
from .video_stream import BoxStabilizer as BoxStabilizer
from .video_stream import LabelColors as LabelColors
from .video_stream import VideoStreamServer as VideoStreamServer
from .video_stream import draw_caption as draw_caption
from .video_stream import draw_crossing_line as draw_crossing_line
from .video_stream import draw_detections as draw_detections
