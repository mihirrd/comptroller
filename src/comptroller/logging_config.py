"""Logging: append to logs.txt for this project's .py sources only; no console."""

from __future__ import annotations

import logging
from pathlib import Path

LOG_FILE = Path(__file__).resolve().parent / "logs.txt"
_PROJECT_ROOT = Path(__file__).resolve().parent

_LOG_FORMAT = (
    "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(name)s: %(message)s"
)

# Local dirs under the repo that are not first-party source.
_SKIP_TOP = frozenset({".venv", "venv", "__pycache__", ".pytest_cache"})


class _ProjectSourceFilter(logging.Filter):
    """Keep records from .py files under the project root (excludes deps / venv)."""

    def filter(self, record: logging.LogRecord) -> bool:
        path = getattr(record, "pathname", None)
        if not path:
            return False
        try:
            p = Path(path).resolve()
            rel = p.relative_to(_PROJECT_ROOT)
        except (OSError, ValueError):
            return False
        if rel.parts and rel.parts[0] in _SKIP_TOP:
            return False
        return p.suffix == ".py"


def _clear_non_root_handlers() -> None:
    """Remove handlers from all non-root loggers (deps attach Stream/Rich handlers here)."""
    root = logging.getLogger()
    for name in list(logging.Logger.manager.loggerDict.keys()):
        if not isinstance(name, str):
            continue
        log = logging.getLogger(name)
        if log is root:
            continue
        log.handlers.clear()


_patch_installed = False


def _only_root_may_have_handlers() -> None:
    """Ignore addHandler on non-root loggers so libraries cannot attach console/file sinks later."""
    global _patch_installed
    if _patch_installed:
        return
    _orig = logging.Logger.addHandler

    def _wrapped(self: logging.Logger, handler: logging.Handler) -> None:
        if self is not logging.getLogger():
            return
        return _orig(self, handler)

    logging.Logger.addHandler = _wrapped  # type: ignore[method-assign]
    _patch_installed = True


def configure_logging() -> None:
    _only_root_may_have_handlers()

    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.DEBUG)

    fh = logging.FileHandler(LOG_FILE, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(_LOG_FORMAT))
    fh.addFilter(_ProjectSourceFilter())
    root.addHandler(fh)

    logging.lastResort = logging.NullHandler()
    _clear_non_root_handlers()
