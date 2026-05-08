from __future__ import annotations

import logging
import time
from typing import TypedDict

logger = logging.getLogger(__name__)


class PhaseTiming(TypedDict):
    """A dictionary representing the timing of a single phase."""

    name: str
    start: float
    end: float


class TimingContext:
    """A context manager to track the duration of a single phase."""

    def __init__(self, timer: TimingStack, name: str):
        self.timer = timer
        self.name = name
        self.start = 0.0

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        end = time.time()
        self.timer.add_phase(self.name, self.start, end)


class TimingStack:
    """A context manager to track the timing of different phases of evaluation.

    Example:
        timing = TimingStack()
        with timing("load data"):
            dataset = load_data()
        with timing("encode"):
            embeddings = model.encode(dataset)

        timing.quick_plot()
    """

    def __init__(self):
        self.phases: list[PhaseTiming] = []
        self._start_time: float | None = None

    def __call__(self, name: str) -> TimingContext:
        """Returns a TimingContext for the specified phase name."""
        if self._start_time is None:
            self._start_time = time.time()
        return TimingContext(self, name)

    def add_phase(self, name: str, start: float, end: float):
        """Adds a phase timing record."""
        if self._start_time is None:
            self._start_time = start
        self.phases.append(
            {
                "name": name,
                "start": start - self._start_time,
                "end": end - self._start_time,
            }
        )

    def quick_plot(self):
        """Plots a text-based bar chart of the recorded timing phases."""
        if not self.phases:
            logger.info("No timing phases recorded.")
            return

        total_time = max(p["end"] for p in self.phases)
        if total_time == 0:
            return

        bar_length = 50
        max_name_len = max(len(p["name"]) for p in self.phases)

        for phase in self.phases:
            name = phase["name"]
            start = phase["start"]
            end = phase["end"]
            duration = end - start

            start_pos = int((start / total_time) * bar_length)
            end_pos = int((end / total_time) * bar_length)
            bar_len = max(1, end_pos - start_pos)

            bar = (
                " " * start_pos
                + "█" * bar_len
                + " " * (bar_length - start_pos - bar_len)
            )
            logger.info(f"{name:<{max_name_len}} |{bar}| {duration:.1f}s")

        tracked_time = sum(p["end"] - p["start"] for p in self.phases)
        untracked = total_time - tracked_time
        logger.info(
            f"{' ' * max_name_len}  {total_time:.1f}s (untracked: {untracked:.1f}s)"
        )
