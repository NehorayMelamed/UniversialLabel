import logging
import os
from datetime import datetime
from colorlog import ColoredFormatter

from common.general_parameters import BASE_PROJECT_DIRECTORY_PATH


def get_logger(name: str, log_dir: str = os.path.join(BASE_PROJECT_DIRECTORY_PATH, "logs")) -> logging.Logger:
    """
    Create and configure a logger with both colored console and file handlers.

    Args:
        name (str): The name of the logger (usually `__name__`).
        log_dir (str): The directory to save log files.

    Returns:
        logging.Logger: Configured logger instance.
    """
    # Ensure the log directory exists
    os.makedirs(log_dir, exist_ok=True)

    # Create a unique log file for each run
    log_filename = os.path.join(log_dir, f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log")

    # Create a logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)  # Set global logging level

    # Prevent duplication of handlers
    if not logger.hasHandlers():
        # Create console handler with colored output
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)  # Console handler level

        # Create file handler for plain text logging
        file_handler = logging.FileHandler(log_filename)
        file_handler.setLevel(logging.DEBUG)  # File handler level

        # Define formatter for console with colors
        color_formatter = ColoredFormatter(
            fmt="%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            log_colors={
                "DEBUG": "cyan",
                "INFO": "green",
                "WARNING": "yellow",
                "ERROR": "red",
                "CRITICAL": "bold_red",
            },
        )

        # Define formatter for file (no colors)
        file_formatter = logging.Formatter(
            fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

        # Set formatters
        console_handler.setFormatter(color_formatter)
        file_handler.setFormatter(file_formatter)

        # Add handlers to the logger
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

    return logger
