"""EventLogger Main Class

Provides a concise API for logging various events
"""

import atexit
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from .models import (
    BaseEvent,
    BenchmarkChangeEvent,
    FilterChangeEvent,
    PageViewEvent,
    TableDownloadEvent,
    TableSwitchEvent,
)
from .storage import MongoDBStorage

logger = logging.getLogger(__name__)


class EventLogger:
    """Event logger main class

    Features:
    - Uses thread pool for asynchronous writes, non-blocking main thread
    - Silent failure, does not affect main business logic
    - Supports all predefined event types
    - Supports custom events

    Usage example:
        ```python
        from event_logger import EventLogger

        # Initialize (reads MONGO_URI from environment variable)
        event_logger = EventLogger()

        # Log filter change
        event_logger.log_filter_change(
            session_id="s_xyz789",
            filter_name="task_type",
            new_value="classification",
            benchmark="MTEB-English"
        )
        ```
    """

    def __init__(
        self,
        mongo_uri: str | None = None,
        database: str | None = None,
        collection: str | None = None,
        max_workers: int = 2,
    ):
        """Initialize event logger

        Args:
            mongo_uri: MongoDB connection string, defaults to reading from environment variable MONGO_URI
            database: Database name
            collection: Collection name
            max_workers: Maximum number of worker threads in thread pool
        """
        self._storage: MongoDBStorage | None = None
        self._storage_kwargs = {
            "uri": mongo_uri,
            "database": database,
            "collection": collection,
        }

        # If MONGO_URI is not set, disable event logging and log one WARNING (do not create MongoDBStorage)
        effective_uri = mongo_uri or os.getenv("MONGO_URI")
        self._storage_disabled = not (effective_uri and effective_uri.strip())
        if self._storage_disabled:
            logger.warning(
                "MONGO_URI not configured; event logging disabled. "
                "Set environment variable MONGO_URI or pass mongo_uri to enable."
            )

        # Thread pool for asynchronous writes
        self._executor = ThreadPoolExecutor(max_workers=max_workers)

        # Register cleanup on exit
        atexit.register(self._cleanup)

        self._initialized = False

    def _ensure_storage(self) -> bool:
        """Ensure storage layer is initialized

        Returns:
            Whether initialization was successful
        """
        if self._storage is not None:
            return True
        if self._storage_disabled:
            return False

        try:
            self._storage = MongoDBStorage(**self._storage_kwargs)
            self._initialized = True
            return True
        except Exception as e:
            logger.warning(
                f"EventLogger initialization failed (will silently skip event logging): {e}"
            )
            return False

    def _log_async(self, event: BaseEvent):
        """Asynchronously write event (executed in background thread)

        Args:
            event: Event object
        """
        if not self._ensure_storage():
            return

        try:
            event_data = event.to_mongo_dict()
            self._storage.insert(event_data)
        except Exception as e:
            # Silent failure
            logger.debug(f"Event logging failed: {e}")

    def log(self, event: BaseEvent):
        """Log event (asynchronous, non-blocking)

        Args:
            event: Event object
        """
        self._executor.submit(self._log_async, event)

    def log_raw(
        self,
        event_name: str,
        session_id: str,
        benchmark: str | None = None,
        filters: dict[str, Any] | None = None,
        properties: dict[str, Any] | None = None,
        **kwargs,
    ):
        """Log raw event (generic method)

        Args:
            event_name: Event name
            session_id: Session ID
            benchmark: Benchmark name
            filters: Filter conditions snapshot
            properties: Event properties
            **kwargs: Other custom fields
        """
        event = BaseEvent(
            event_name=event_name,
            session_id=session_id,
            benchmark=benchmark,
            filters=filters,
            properties=properties,
            **kwargs,
        )
        self.log(event)

    # ============ Convenience Methods ============

    def log_page_view(self, session_id: str, benchmark: str | None = None, **kwargs):
        """Log page view event

        Args:
            session_id: Session ID
            benchmark: Current benchmark
            **kwargs (Any): Additional context.
        """
        event = PageViewEvent(session_id=session_id, benchmark=benchmark, **kwargs)
        self.log(event)

    def log_benchmark_change(
        self, session_id: str, new_value: str, old_value: str | None = None, **kwargs
    ):
        """Log benchmark change event

        Args:
            session_id: Session ID
            new_value: Newly selected benchmark
            old_value: Previous benchmark
            **kwargs (Any): Additional context.
        """
        event = BenchmarkChangeEvent.create(
            session_id=session_id, old_value=old_value, new_value=new_value, **kwargs
        )
        self.log(event)

    def log_filter_change(
        self,
        session_id: str,
        filter_name: str,
        new_value: Any,
        old_value: Any = None,
        benchmark: str | None = None,
        filters: dict[str, Any] | None = None,
        **kwargs,
    ):
        """Log filter change event

        Args:
            session_id: Session ID
            filter_name: Filter name (e.g., task_type, domain, language, etc.)
            new_value: New value
            old_value: Old value
            benchmark: Current benchmark
            filters: Snapshot of all current filter conditions
            **kwargs (Any): Additional context.
        """
        event = FilterChangeEvent.create(
            session_id=session_id,
            filter_name=filter_name,
            new_value=new_value,
            old_value=old_value,
            benchmark=benchmark,
            filters=filters,
            **kwargs,
        )
        self.log(event)

    def log_table_switch(
        self,
        session_id: str,
        new_table: str,
        old_table: str | None = None,
        benchmark: str | None = None,
        **kwargs,
    ):
        """Log table switch event

        Args:
            session_id: Session ID
            new_table: New table
            old_table: Old table
            benchmark: Current benchmark
            **kwargs (Any): Additional context.
        """
        event = TableSwitchEvent.create(
            session_id=session_id,
            old_table=old_table,
            new_table=new_table,
            benchmark=benchmark,
            **kwargs,
        )
        self.log(event)

    def log_table_download(
        self,
        session_id: str,
        benchmark: str | None = None,
        format: str = "csv",
        row_count: int | None = None,
        **kwargs,
    ):
        """Log table download event

        Args:
            session_id: Session ID
            benchmark: Current benchmark
            format: Download format (e.g., csv, json)
            row_count: Number of rows downloaded
            **kwargs (Any): Additional context.
        """
        event = TableDownloadEvent.create(
            session_id=session_id,
            benchmark=benchmark,
            format=format,
            row_count=row_count,
            **kwargs,
        )
        self.log(event)

    def _cleanup(self):
        """Clean up resources"""
        self._executor.shutdown(wait=True)
        if self._storage:
            self._storage.close()
