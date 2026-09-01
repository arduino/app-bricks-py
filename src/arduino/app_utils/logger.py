# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

import logging
import os


def _resolve_level(level: int) -> int:
    """Applies the APP_BRICKS_LOG_LEVEL environment override to the given level."""
    override_log_level = os.getenv("APP_BRICKS_LOG_LEVEL")
    if override_log_level is not None:
        return getattr(logging, override_log_level.upper(), logging.INFO)
    return level


def _build_handler() -> logging.Handler:
    """Builds a stream handler with the app-standard log format."""
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        fmt="%(asctime)s.%(msecs)03d %(levelname)s - [%(threadName).32s] %(name)s:  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    return handler


def _configure_library_logger(name: str, display_name: str | None = None, level: int = logging.INFO) -> None:
    """Overrides the named logger's handler, format and log level with the ones used by the Bricks framework.

    Args:
        name (str): The library's logger name, e.g. "arduino.router_bridge".
        display_name (str, optional): Name shown in log records instead of the library's logger name.
        level (int): The logging level, subject to the APP_BRICKS_LOG_LEVEL override. Defaults to logging.INFO.
    """
    handler = _build_handler()
    if display_name is not None:

        def rename(record: logging.LogRecord) -> bool:
            record.name = display_name
            return True

        handler.addFilter(rename)
    lib_logger = logging.getLogger(name)
    lib_logger.handlers = [handler]
    lib_logger.propagate = False
    lib_logger.setLevel(_resolve_level(level))


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

    def __init__(self, name: str, level: int = logging.INFO) -> None:
        super().__init__(name, _resolve_level(level))
        self.handlers = []  # Remove inherited handlers
        self.addHandler(_build_handler())

    def process[T](self, msg: T) -> T:
        self.info(msg)
        return msg

    def consume(self, msg: object) -> None:
        self.info(msg)
