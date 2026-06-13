# logger_config.py

import os
from datetime import datetime

from loguru import logger
from rich.console import Console
from rich.logging import RichHandler

_initialized = False


def get_logger(
    name: str,
    log_dir: str = "logs",
    console_level: str = "DEBUG",
    file_level: str = "DEBUG",
    error_level: str = "ERROR",
    rotation: str = "1 day",
    retention: str = "7 days",
    compression: str = "zip",
    json_logs: bool = False,
    per_run: bool = True,
):
    """
    Configure and return a Loguru logger with:
    - Rich colored console output
    - Per-run log files
    - Separate error logs
    - Optional JSON logs
    """

    global _initialized

    os.makedirs(log_dir, exist_ok=True)

    if not _initialized:
        logger.remove()

        # Rich console for colorful logs
        console = Console()

        logger.add(
            RichHandler(
                console=console,
                rich_tracebacks=True,
                tracebacks_show_locals=True,
                show_path=False,
                show_time=True,
                show_level=True,
                markup=True,
            ),
            level=console_level,
            format="{message}",
        )

        _initialized = True

    # File format
    file_format = (
        "{time:YYYY-MM-DD HH:mm:ss} | "
        "{extra[name]} | "
        "{level:<8} | "
        "{message}\n"
        "{exception}"
    )

    # Main log file
    if per_run:
        run_ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        logger.add(
            os.path.join(log_dir, f"{name}_{run_ts}.log"),
            level=file_level,
            format=file_format,
            retention=retention,
            encoding="utf-8",
        )
    else:
        logger.add(
            os.path.join(log_dir, f"{name}.log"),
            level=file_level,
            format=file_format,
            rotation=rotation,
            retention=retention,
            compression=compression,
            encoding="utf-8",
        )

    # Error log file
    logger.add(
        os.path.join(log_dir, f"{name}_errors.log"),
        level=error_level,
        format=file_format,
        retention="30 days",
        encoding="utf-8",
    )

    # Optional JSON logs
    if json_logs:
        logger.add(
            os.path.join(log_dir, f"{name}_structured.json"),
            level=file_level,
            serialize=True,
            rotation=rotation,
            retention=retention,
            encoding="utf-8",
        )

    return logger.bind(name=name)
