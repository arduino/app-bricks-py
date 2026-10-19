# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""Pytest configuration for the aihub-onnx-models-runner container tests.

The tests run in this image's venv, the one its runners run in: they cover the ``aihub`` framework
the image adds and the ocr-runner built on it. Adds the framework directory to ``sys.path`` so
``import aihub`` resolves the way the image's ``PYTHONPATH`` makes it resolve at runtime.
"""

import os
import sys

FRAMEWORK_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "base", "aihub-framework"))
if FRAMEWORK_DIR not in sys.path:
    sys.path.insert(0, FRAMEWORK_DIR)
