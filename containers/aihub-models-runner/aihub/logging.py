# SPDX-FileCopyrightText: Copyright (C) ARDUINO SRL (http://www.arduino.cc)
#
# SPDX-License-Identifier: MPL-2.0

import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger()


def setup_logging(verbose: bool = False) -> None:
    """Configure aihub logging. Call once at startup."""
    level = logging.DEBUG if verbose else logging.INFO
    logger.setLevel(level)
