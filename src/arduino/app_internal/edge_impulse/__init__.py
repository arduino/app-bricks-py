# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""Base classes of the bricks running Edge Impulse models on the `arduino:edge_impulse` inference service.

`EdgeImpulseModel` holds the model and its connection for any input. `VideoInference` in the `video` module runs it
on the frames of a camera and serves the video with an overlay; the video bricks of the library build on it, and so
can yours. The video modules are imported on their own so a brick on `EdgeImpulseModel` alone does not load them.
"""

from .model import EdgeImpulseModel as EdgeImpulseModel
