import logging
import config
from colorlog import ColoredFormatter

def configure_logger():
    """Configure and return a logger with colored output."""
    logger = logging.getLogger("zkGitTestLogger")
    logger.setLevel(config.LOGGING_LEVEL)

    # Define log format with color
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

    # Create a stream handler (console output)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    # Add the handler to the logger
    logger.addHandler(console_handler)

    return logger

