# Copyright 2025-2026 Dimensional Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from datetime import datetime
import inspect
import logging
import logging.handlers
import os
from pathlib import Path
import sys
import tempfile
import traceback
from typing import Any, Mapping, Optional

import structlog
from structlog.processors import CallsiteParameter, CallsiteParameterAdder

from dimos.constants import DIMOS_LOG_DIR, DIMOS_PROJECT_ROOT

# Suppress noisy loggers
logging.getLogger("aiortc.codecs.h264").setLevel(logging.ERROR)
logging.getLogger("lcm_foxglove_bridge").setLevel(logging.ERROR)
logging.getLogger("websockets.server").setLevel(logging.ERROR)
logging.getLogger("FoxgloveServer").setLevel(logging.ERROR)
logging.getLogger("asyncio").setLevel(logging.ERROR)

_LOG_FILE_PATH: Path | None = None
_LOG_FILE_WARNING_EMITTED = False


def _candidate_log_directories() -> list[Path]:
    """Return preferred log directories ordered by priority."""

    candidates: list[Path] = []

    env_override = os.getenv("DIMOS_LOG_DIR")
    if env_override:
        candidates.append(Path(env_override))

    candidates.append(DIMOS_LOG_DIR)

    xdg_state_home = os.getenv("XDG_STATE_HOME")
    if xdg_state_home:
        candidates.append(Path(xdg_state_home) / "dimos" / "logs")

    try:
        home = Path.home()
    except RuntimeError:
        home = None

    if home is not None:
        candidates.append(home / ".cache" / "dimos" / "logs")

    candidates.append(Path(tempfile.gettempdir()) / "dimos" / "logs")

    # Remove duplicates while preserving order
    seen = set()
    unique_candidates = []
    for candidate in candidates:
        if candidate not in seen:
            unique_candidates.append(candidate)
            seen.add(candidate)

    return unique_candidates


def _ensure_writable_directory(directory: Path) -> bool:
    """Ensure the directory exists and is writable."""

    try:
        directory.mkdir(parents=True, exist_ok=True)
    except PermissionError:
        return False
    except OSError:
        return False

    return os.access(directory, os.W_OK)


def _get_log_file_path() -> Path | None:
    for directory in _candidate_log_directories():
        if _ensure_writable_directory(directory):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            pid = os.getpid()
            return directory / f"dimos_{timestamp}_{pid}.jsonl"

    return None


def _configure_structlog() -> Path | None:
    global _LOG_FILE_PATH

    if _LOG_FILE_PATH:
        return _LOG_FILE_PATH

    _LOG_FILE_PATH = _get_log_file_path()

    shared_processors = [
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
        CallsiteParameterAdder(
            parameters=[
                CallsiteParameter.FUNC_NAME,
                CallsiteParameter.LINENO,
            ]
        ),
        structlog.processors.format_exc_info,  # Add this to format exception info
    ]

    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    return _LOG_FILE_PATH


def setup_logger(level: int | str | None = None, *, name: str | None = None) -> Any:
    """Set up a structured logger using structlog.

    Args:
        level: The logging level.

    Returns:
        A configured structlog logger instance.
    """

    caller_frame = inspect.stack()[1]
    inferred_name = caller_frame.filename

    # Convert absolute path to relative path
    try:
        inferred_name = str(Path(inferred_name).relative_to(DIMOS_PROJECT_ROOT))
    except (ValueError, TypeError):
        pass

    global _LOG_FILE_WARNING_EMITTED

    log_file_path = _configure_structlog()

    if log_file_path is None and not _LOG_FILE_WARNING_EMITTED:
        sys.stderr.write(
            "Warning: Unable to create a writable log directory. File logging will be disabled.\n"
        )
        _LOG_FILE_WARNING_EMITTED = True

    configured_name = name
    configured_level = level

    if isinstance(level, str):
        normalized_level = level.upper()
        if hasattr(logging, normalized_level):
            configured_level = getattr(logging, normalized_level)
        else:
            configured_name = level
            configured_level = None

    if configured_name is None:
        configured_name = inferred_name

    if configured_level is None:
        level_name = os.getenv("DIMOS_LOG_LEVEL", "INFO").upper()
        configured_level = getattr(logging, level_name, logging.INFO)

    stdlib_logger = logging.getLogger(configured_name)

    # Remove any existing handlers.
    if stdlib_logger.hasHandlers():
        stdlib_logger.handlers.clear()

    stdlib_logger.setLevel(configured_level)
    stdlib_logger.propagate = False

    # Create console handler with pretty formatting.
    # We use exception_formatter=None because we handle exceptions
    # separately with Rich in the global exception handler
    console_renderer = structlog.dev.ConsoleRenderer(
        colors=True,
        pad_event=60,
        force_colors=False,
        sort_keys=True,
        exception_formatter=None,  # Don't format exceptions in console logs
    )

    # Wrapper to remove callsite info and exception details before rendering to console.
    def console_processor_without_callsite(
        logger: Any, method_name: str, event_dict: Mapping[str, Any]
    ) -> str:
        event_dict = dict(event_dict)
        # Remove callsite info
        event_dict.pop("func_name", None)
        event_dict.pop("lineno", None)
        # Remove exception fields since we handle them with Rich
        event_dict.pop("exception", None)
        event_dict.pop("exc_info", None)
        event_dict.pop("exception_type", None)
        event_dict.pop("exception_message", None)
        event_dict.pop("traceback_lines", None)
        return console_renderer(logger, method_name, event_dict)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(configured_level)
    console_formatter = structlog.stdlib.ProcessorFormatter(
        processor=console_processor_without_callsite,
    )
    console_handler.setFormatter(console_formatter)
    stdlib_logger.addHandler(console_handler)

    # Create rotating file handler with JSON formatting.
    if log_file_path:
        file_handler = logging.handlers.RotatingFileHandler(
            log_file_path,
            mode="a",
            maxBytes=10 * 1024 * 1024,  # 10MiB
            backupCount=20,
            encoding="utf-8",
        )
        file_handler.setLevel(configured_level)
        file_formatter = structlog.stdlib.ProcessorFormatter(
            processor=structlog.processors.JSONRenderer(),
        )
        file_handler.setFormatter(file_formatter)
        stdlib_logger.addHandler(file_handler)

    return structlog.get_logger(configured_name)


def setup_exception_handler() -> None:
    def handle_exception(exc_type, exc_value, exc_traceback):
        # Don't log KeyboardInterrupt
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return

        # Get a logger for uncaught exceptions
        logger = setup_logger("uncaught_exception")

        # Log the exception with full traceback to JSON
        logger.error(
            "Uncaught exception occurred",
            exc_info=(exc_type, exc_value, exc_traceback),
            exception_type=exc_type.__name__,
            exception_message=str(exc_value),
            traceback_lines=traceback.format_exception(exc_type, exc_value, exc_traceback),
        )

        # Still display the exception nicely on console using Rich if available
        try:
            from rich.console import Console
            from rich.traceback import Traceback

            console = Console()
            tb = Traceback.from_exception(exc_type, exc_value, exc_traceback)
            console.print(tb)
        except ImportError:
            # Fall back to standard exception display if Rich is not available
            sys.__excepthook__(exc_type, exc_value, exc_traceback)

    # Set our custom exception handler
    sys.excepthook = handle_exception
