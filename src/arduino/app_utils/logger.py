# SPDX-FileCopyrightText: Copyright (C) ARDUINO SRL (http://www.arduino.cc)
#
# SPDX-License-Identifier: MPL-2.0

import logging
import os


class Logger(logging.Logger):
    """A simple logger class that extends Python's logging.Logger.
    Log levels can also be customized using the APP_BRICKS_LOG_LEVEL environment variable (FATAL, CRITICAL, ERROR, WARNING, INFO, DEBUG).

    Args:
        name (str): The name of the logger. You can use the dot syntax (parent.child) to create a hierarchy of loggers.
        level (int or str): The logging level. Defaults to logging.WARNING.

    Examples:
        logger = Logger('my_logger')
        logger.error('This is an error message and will be printed')
        logger.warning('This is a warning message and will be printed')
        logger.info('This is an info message and won't be printed by default')
        logger.debug('This is a debug message and won't be printed by default')
        logger.print('This will always be printed, regardless of the level')
    """

    def __init__(self, name: str, level: int = logging.INFO):
        override_log_level = os.getenv("APP_BRICKS_LOG_LEVEL")
        if override_log_level is not None:
            level = getattr(logging, override_log_level.upper(), logging.INFO)

        super().__init__(name, level)
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            fmt="%(asctime)s.%(msecs)03d %(levelname)s - [%(threadName).32s] %(name)s:  %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        self.handlers = []  # Remove inherited handlers
        self.addHandler(handler)

    def process(self, msg):
        self.info(msg)
        return msg

    def consume(self, msg):
        self.info(msg)
