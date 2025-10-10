from typing import ClassVar


class LogOnce:
    """Logger wrapper that ensures each unique message is logged only once per logger instance."""

    _seen: ClassVar[dict] = {}  # Class-level cache shared across instances

    def __init__(self, logger):
        self.logger = logger
        # Use logger name as key to have separate caches per logger
        if logger.name not in self._seen:
            self._seen[logger.name] = set()

    def info(self, msg):
        if msg not in self._seen[self.logger.name]:
            self.logger.info(msg)
            self._seen[self.logger.name].add(msg)

    def warning(self, msg):
        if msg not in self._seen[self.logger.name]:
            self.logger.warning(msg)
            self._seen[self.logger.name].add(msg)

    def error(self, msg):
        if msg not in self._seen[self.logger.name]:
            self.logger.error(msg)
            self._seen[self.logger.name].add(msg)

    def debug(self, msg):
        if msg not in self._seen[self.logger.name]:
            self.logger.debug(msg)
            self._seen[self.logger.name].add(msg)
