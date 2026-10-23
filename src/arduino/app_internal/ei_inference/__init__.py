# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""Client of the Edge Impulse inference service, the edge-impulse-runner and edge-impulse-npu-runner containers.

The service serves the ``.eim`` models over one Unix socket, one connection per model. This package holds the
socket protocol and the client the bricks build on. Frames go out as captured, the service resizes them to the
model input and returns the boxes in the frame coordinates.
"""

from .client import DEFAULT_SOCKET_PATH as DEFAULT_SOCKET_PATH
from .client import Box as Box
from .client import InferenceClient as InferenceClient
from .client import Result as Result
from .client import ServerError as ServerError
from .client import model_name_from_path as model_name_from_path
from .protocol import ProtocolError as ProtocolError
