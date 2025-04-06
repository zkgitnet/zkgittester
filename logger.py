"""
Logger Configuration Module

This module sets up and returns a configured logger for use in the zkGit testing framework.
It utilizes `colorlog` to produce colored console output, making it easier to distinguish
between log levels such as DEBUG, INFO, WARNING, ERROR, and CRITICAL.

The logger configuration is based on the global logging level defined in `config.LOGGING_LEVEL`.
"""

import logging
from colorlog import ColoredFormatter
import config

def configure_logger():
    """Configure and return a logger with colored output."""
    logger = logging.getLogger("zkGitTestLogger")
    logger.setLevel(config.LOGGING_LEVEL)

    formatter = ColoredFormatter(
        '%(log_color)s%(asctime)s %(funcName)-20s %(levelname)-8s%(reset)s %(message)s',
        log_colors={
            'DEBUG': 'cyan',
            'INFO': 'green',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'magenta',
        }
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger
