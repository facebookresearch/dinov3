# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

import functools
import logging
import os
import sys
from typing import Optional

from termcolor import colored

from dinov3.distributed import TorchDistributedEnvironment

from dinov3.logging.helpers import MetricLogger, SmoothedValue

_LEVEL_COLORED_KWARGS = {
    logging.DEBUG: {"color": "green", "attrs": ["bold"]},
    logging.INFO: {"color": "green"},
    logging.WARNING: {"color": "yellow"},
    logging.ERROR: {"color": "red"},
    logging.CRITICAL: {"color": "red", "attrs": ["bold"]},
}


class _LevelColoredFormatter(logging.Formatter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def formatMessage(self, record):
        log = super().formatMessage(record)

        colored_kwargs = _LEVEL_COLORED_KWARGS.get(record.levelno)
        if colored_kwargs is None:
            return log

        raw_msg = record.getMessage()
        msg = str(raw_msg)
        
        index = log.rfind(msg, len(log) - len(msg))
        # Can happen in some cases, like if the msg contains `%s` which
        # have been replaced in `formatMessage`. Fallback to no colors
        if index == -1:
            return log
        prefix = log[:index]
        prefix = colored(prefix, **colored_kwargs)
        return prefix + msg


# So that calling _configure_logger multiple times won't add many handlers
@functools.lru_cache()
def _configure_logger(
    name: Optional[str] = None,
    *,
    level: int = logging.DEBUG,
    output: Optional[str] = None,
    color: bool = True,
    log_to_stdout_only_in_main_process: bool = True,
):
    """
    Configure a logger.

    Adapted from Detectron2.

    Args:
        name: The name of the logger to configure.
        level: The logging level to use.
        output: A file name or a directory to save log. If None, will not save log file.
            If ends with ".txt" or ".log", assumed to be a file name.
            Otherwise, logs will be saved to `output/log.txt`.
        color: Whether stdout output should be colored (ignored if stdout is not a terminal).
        log_to_stdout_only_in_main_process: The main process (rank 0) always logs to stdout,
            regardless of this flag. If False, other ranks will also log to their stdout.

    Returns:
        The configured logger.
    """

    # Disable colored output if the stdout is not a terminal
    color = color and os.isatty(sys.stdout.fileno())

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    # Loosely match Google glog format:
    #   [IWEF]yyyymmdd hh:mm:ss.uuuuuu threadid file:line] msg
    # but use a shorter timestamp and include the logger name:
    #   [IWEF]yyyymmdd hh:mm:ss logger threadid file:line] msg
    fmt_prefix = "%(levelname).1s%(asctime)s %(process)s %(name)s %(filename)s:%(lineno)s] "
    fmt_message = "%(message)s"
    fmt = fmt_prefix + fmt_message
    datefmt = "%Y%m%d %H:%M:%S"
    plain_formatter = logging.Formatter(fmt=fmt, datefmt=datefmt)

    torch_env = TorchDistributedEnvironment()

    # rank 0 always logs to stdout, for other ranks it depends on log_to_stdout_only_in_main_process
    should_log_to_stdout = torch_env.is_main_process or not log_to_stdout_only_in_main_process
    if should_log_to_stdout:
        handler = logging.StreamHandler(stream=sys.stdout)
        handler.setLevel(logging.DEBUG)

        formatter: logging.Formatter
        if color:
            formatter = _LevelColoredFormatter(
                fmt=fmt,
                datefmt=datefmt,
            )
        else:
            formatter = plain_formatter

        handler.setFormatter(formatter)
        logger.addHandler(handler)

    # file logging for all workers
    if output:
        if os.path.splitext(output)[-1] in (".txt", ".log"):
            filename = output
        else:
            filename = os.path.join(output, "logs", "log.txt")

        if not torch_env.is_main_process:
            filename = filename + f".rank{torch_env.rank}"

        os.makedirs(os.path.dirname(filename), exist_ok=True)

        handler = logging.StreamHandler(open(filename, "a"))
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(plain_formatter)
        logger.addHandler(handler)

    logger.debug(f"PyTorch distributed environment: {torch_env}")
    return logger


def setup_logging(
    output: Optional[str] = None,
    *,
    name: Optional[str] = None,
    level: int = logging.DEBUG,
    color: bool = True,
    capture_warnings: bool = True,
    log_to_stdout_only_in_main_process: bool = True,
) -> None:
    """
    Setup logging.

    Args:
        output: A file name or a directory to save log files. If None, log
            files will not be saved. If output ends with ".txt" or ".log", it
            is assumed to be a file name.
            Otherwise, logs will be saved to `output/log.txt`.
        name: The name of the logger to configure, by default the root logger.
        level: The logging level to use.
        color: Whether stdout output should be colored (ignored if stdout is not a terminal).
        capture_warnings: Whether warnings should be captured as logs.
        log_to_stdout_only_in_main_process: The main process (rank 0) always logs to stdout,
            regardless of this flag. If False, other ranks will also log to their stdout.
    """
    logging.captureWarnings(capture_warnings)
    # Ensure the path is canonical to properly use the cache of `_configure_logger`
    output = output if output is None else os.path.realpath(output)
    _configure_logger(
        name,
        level=level,
        output=output,
        color=color,
        log_to_stdout_only_in_main_process=log_to_stdout_only_in_main_process,
    )


def cleanup_logging(*, name: Optional[str] = None) -> None:
    logger = logging.getLogger(name)
    for handler in logger.handlers:
        handler.flush()
        handler.close()
        logger.removeHandler(handler)

    # clears the cache of `_configure_logger` to allow re-initialization
    _configure_logger.cache_clear()
