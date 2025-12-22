"""Specific Event Type Models

Defines data structures for various business events
"""

from typing import Any, Literal

from pydantic import Field

from .base import BaseEvent

# Supported filter types
FilterName = Literal[
    "task_type",
    "domain",
    "modality",
    "language",
    "task",
    "compatibility",
    "availability",
    "instructions",
    "zero-shot",
    "modelParameters",
]


class PageViewEvent(BaseEvent):
    """Page view event

    Triggered when user visits the page
    """

    event_name: str = Field(default="page_view", frozen=True)


class BenchmarkChangeEvent(BaseEvent):
    """Benchmark change event

    Triggered when user switches benchmark in sidebar
    """

    event_name: str = Field(default="benchmark_change", frozen=True)
    properties: dict[str, Any] = Field(
        default_factory=dict, description="Contains old_value and new_value"
    )

    @classmethod
    def create(
        cls, session_id: str, old_value: str | None, new_value: str, **kwargs
    ) -> "BenchmarkChangeEvent":
        """Convenience creation method"""
        return cls(
            session_id=session_id,
            benchmark=new_value,
            properties={
                "old_value": old_value,
                "new_value": new_value,
            },
            **kwargs,
        )


class FilterChangeEvent(BaseEvent):
    """Filter change event

    Triggered when user modifies any filter condition
    event_name format: filter_change_{filter_name}
    """

    event_name: str = Field(..., description="filter_change_{filter_name}")
    properties: dict[str, Any] = Field(
        default_factory=dict, description="Contains filter_name, old_value, new_value"
    )

    @classmethod
    def create(
        cls,
        session_id: str,
        filter_name: str,
        new_value: Any,
        old_value: Any = None,
        benchmark: str | None = None,
        filters: dict[str, Any] | None = None,
        **kwargs,
    ) -> "FilterChangeEvent":
        """Convenience creation method"""
        return cls(
            event_name=f"filter_change_{filter_name}",
            session_id=session_id,
            benchmark=benchmark,
            filters=filters,
            properties={
                "filter_name": filter_name,
                "old_value": old_value,
                "new_value": new_value,
            },
            **kwargs,
        )


class TableSwitchEvent(BaseEvent):
    """Table switch event

    Triggered when user switches between different table views
    """

    event_name: str = Field(default="table_switch", frozen=True)
    properties: dict[str, Any] = Field(
        default_factory=dict, description="Contains old_table and new_table"
    )

    @classmethod
    def create(
        cls,
        session_id: str,
        old_table: str | None,
        new_table: str,
        benchmark: str | None = None,
        **kwargs,
    ) -> "TableSwitchEvent":
        """Convenience creation method"""
        return cls(
            session_id=session_id,
            benchmark=benchmark,
            properties={
                "old_table": old_table,
                "new_table": new_table,
            },
            **kwargs,
        )


class TableDownloadEvent(BaseEvent):
    """Table download event

    Triggered when user downloads table data
    """

    event_name: str = Field(default="table_download", frozen=True)
    properties: dict[str, Any] = Field(
        default_factory=dict,
        description="Contains download-related information, such as format, row_count, etc.",
    )

    @classmethod
    def create(
        cls,
        session_id: str,
        benchmark: str | None = None,
        format: str = "csv",
        row_count: int | None = None,
        **kwargs,
    ) -> "TableDownloadEvent":
        """Convenience creation method"""
        props = {"format": format}
        if row_count is not None:
            props["row_count"] = row_count

        return cls(
            session_id=session_id, benchmark=benchmark, properties=props, **kwargs
        )
