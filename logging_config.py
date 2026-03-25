"""Application logging: file-only output, console handlers stripped from deps."""

from __future__ import annotations

import logging
from pathlib import Path

LOG_FILE = Path(__file__).resolve().parent / "logs.txt"

_LOG_FORMAT = (
    "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(name)s: %(message)s"
)


def configure_logging() -> None:
    """Route logs to LOG_FILE only (not stderr)."""
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter(_LOG_FORMAT)
    file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(fmt)

    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.DEBUG)
    root.addHandler(file_handler)


def detach_console_handlers() -> None:
    """Remove stderr/stdout handlers from root and registered named loggers.

    Libraries attach StreamHandlers to their own loggers; without this, duplicate
    output can appear on the terminal while records still propagate to the file.
    """
    names = [""] + list(logging.root.manager.loggerDict.keys())
    seen: set[int] = set()
    for name in names:
        log = logging.getLogger(name)
        if id(log) in seen:
            continue
        seen.add(id(log))
        if not isinstance(log, logging.Logger):
            continue
        for h in list(log.handlers):
            if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler):
                log.removeHandler(h)
