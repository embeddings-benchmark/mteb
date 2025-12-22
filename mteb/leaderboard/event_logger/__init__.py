"""Event Logger - Event Logger

A lightweight event logging SDK for Gradio projects

Features:
- Respects user privacy, does not record any user information
- Uses MongoDB for storage
- Asynchronous writes, non-blocking main thread
- Silent failure, does not affect main business logic
- Concise API design

Usage example:
    ```python
    import gradio as gr
    from event_logger import EventLogger, get_session_js

    # Initialize (requires MONGO_URI environment variable)
    logger = EventLogger()

    with gr.Blocks() as demo:
        session_id = gr.State()
        benchmark_select = gr.Dropdown(choices=["MTEB-English", "MTEB-Chinese"])
        type_select = gr.Dropdown(choices=["all", "classification", "clustering"])

        # Initialize session_id
        demo.load(
            fn=lambda x: x,
            inputs=[],
            outputs=[session_id],
            js=get_session_js()
        )

        # Log events in callbacks
        def on_type_change(type_val, benchmark, session_id):
            # Your business logic...

            # Log event
            logger.log_filter_change(
                session_id=session_id,
                filter_name="task_type",
                new_value=type_val,
                benchmark=benchmark
            )
            return type_val

        type_select.change(
            on_type_change,
            inputs=[type_select, benchmark_select, session_id],
            outputs=[type_select]
        )
    ```
"""

from .logger import EventLogger
from .models import (
    BaseEvent,
    BenchmarkChangeEvent,
    FilterChangeEvent,
    PageViewEvent,
    TableDownloadEvent,
    TableSwitchEvent,
)

__version__ = "0.1.0"

__all__ = [
    "BaseEvent",
    "BenchmarkChangeEvent",
    "EventLogger",
    "FilterChangeEvent",
    "PageViewEvent",
    "TableDownloadEvent",
    "TableSwitchEvent",
]
