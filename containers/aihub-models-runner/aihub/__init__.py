# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

from aihub.app import AIHubApp
from aihub.cli import parse_args, create_parser
from aihub.base import InputSource, OutputSink

__all__ = [
    "AIHubApp",
    "parse_args",
    "create_parser",
    "InputSource",
    "OutputSink",
]

__version__ = "0.1.0"
