"""
Thin logging wrapper that replicates the WiseryLogger interface used
throughout the original AppServer codebase.
"""
import logging
import sys

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def _configure_root() -> None:
    if not logging.root.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))
        logging.root.addHandler(handler)
        logging.root.setLevel(logging.INFO)


_configure_root()


class WiseryLogger:
    """Drop-in replacement for AppServer's WiseryLogger."""

    def __init__(self, name: str = "wiki-retrieval"):
        self._logger = logging.getLogger(name)

    def get_logger(self) -> "WiseryLogger":
        return self

    # ---- standard levels ----
    def info(self, msg: str, *args, **kwargs) -> None:
        self._logger.info(msg, *args, **kwargs)

    def debug(self, msg: str, *args, **kwargs) -> None:
        self._logger.debug(msg, *args, **kwargs)

    def warning(self, msg: str, *args, **kwargs) -> None:
        self._logger.warning(msg, *args, **kwargs)

    def error(self, msg: str, *args, **kwargs) -> None:
        self._logger.error(msg, *args, **kwargs)

    def exception(self, msg: str, *args, **kwargs) -> None:
        self._logger.exception(msg, *args, **kwargs)


def get_logger(name: str = "wiki-retrieval") -> WiseryLogger:
    """Return a WiseryLogger instance for the given name."""
    return WiseryLogger(name)
