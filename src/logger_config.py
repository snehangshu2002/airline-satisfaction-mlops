# logger_config.py

import sys
import os
from datetime import datetime
from loguru import logger

_initialized = False  # guard — ensure logger.remove() runs only once


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
    per_run: bool = True   # new — one file per run instead of appending
):
    """
    Returns a configured loguru logger.

    Args:
        name          : Name for the log file (e.g. 'data_ingestion')
        log_dir       : Directory to store log files (default: 'logs')
        console_level : Minimum level for console output (default: 'DEBUG')
        file_level    : Minimum level for .log file (default: 'DEBUG')
        error_level   : Minimum level for errors-only file (default: 'ERROR')
        rotation      : When to rotate log file (default: '1 day')
        retention     : How long to keep old logs (default: '7 days')
        compression   : Compression format for rotated logs (default: 'zip')
        json_logs     : Whether to also write structured JSON logs (default: False)
        per_run       : Create a new log file for each run (default: True)
    """
    global _initialized

    os.makedirs(log_dir, exist_ok=True)

    # Remove default sink only once across all modules
    if not _initialized:
        logger.remove()
        _initialized = True

    # Format using a callable — avoids KeyError when extra[name] is missing
    def make_fmt(record):
        record["extra"].setdefault("name", "root")
        return "{time:YYYY-MM-DD HH:mm:ss} - {extra[name]} - {level} - {message}\n{exception}"

    # Console sink
    logger.add(
        sys.stdout,
        level=console_level,
        format=make_fmt,
        colorize=False   # colorize=True breaks when format is a callable
    )

    # Main log file — per run or appending
    if per_run:
        run_ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_filename = os.path.join(log_dir, f"{name}_{run_ts}.log")
        logger.add(
            log_filename,
            level=file_level,
            format=make_fmt,
            retention=retention   # rotation not needed for per-run files
        )
    else:
        logger.add(
            os.path.join(log_dir, f"{name}.log"),
            level=file_level,
            format=make_fmt,
            rotation=rotation,
            retention=retention,
            compression=compression
        )

    # Errors-only file — always appends across runs, useful for history
    logger.add(
        os.path.join(log_dir, f"{name}_errors.log"),
        level=error_level,
        format=make_fmt,
        retention="30 days"
    )

    # Optional JSON structured logs
    if json_logs:
        logger.add(
            os.path.join(log_dir, f"{name}_structured.json"),
            level=file_level,
            serialize=True,
            rotation=rotation,
            retention=retention
        )

    return logger.bind(name=name)