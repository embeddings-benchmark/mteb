from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    from collections.abc import MutableMapping
    from logging import Logger


class LogOnce:
    """Logger wrapper that ensures each unique message is logged only once per logger instance."""

    _seen: ClassVar[
        MutableMapping[str, set[str]]
    ] = {}  # Class-level cache shared across instances

    def __init__(self, logger: Logger) -> None:
        self.logger = logger
        # Use logger name as key to have separate caches per logger
        if logger.name not in self._seen:
            self._seen[logger.name] = set()

    def info(self, msg: str) -> None:
        if msg not in self._seen[self.logger.name]:
            self.logger.info(msg)
            self._seen[self.logger.name].add(msg)

    def warning(self, msg: str) -> None:
        """Log and emit a warnings.warn exactly once per unique message."""
        if msg not in self._seen[self.logger.name]:
            self.logger.warning(msg)
            warnings.warn(msg, stacklevel=2)
            self._seen[self.logger.name].add(msg)

    def error(self, msg: str) -> None:
        if msg not in self._seen[self.logger.name]:
            self.logger.error(msg)
            self._seen[self.logger.name].add(msg)

    def debug(self, msg: str) -> None:
        if msg not in self._seen[self.logger.name]:
            self.logger.debug(msg)
            self._seen[self.logger.name].add(msg)
