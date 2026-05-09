from __future__ import annotations

import logging
import time
from typing import TypedDict

logger = logging.getLogger(__name__)


class PhaseTiming(TypedDict, total=False):
    """A dictionary representing the timing of a single phase."""

    name: str
    start: float
    end: float
    split: str
    subset: str


class TimingContext:
    """A context manager for timing a specific phase."""

    def __init__(
        self,
        stack: TimingStack,
        name: str,
        split: str | None = None,
        subset: str | None = None,
    ):
        self.stack = stack
        self.name = name
        self.split = split
        self.subset = subset
        self.start = 0.0

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        end = time.time()
        self.stack.add_phase(
            self.name, self.start, end, split=self.split, subset=self.subset
        )


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

    def __call__(
        self, name: str, split: str | None = None, subset: str | None = None
    ) -> TimingContext:
        """Returns a TimingContext for the specified phase name."""
        if self._start_time is None:
            self._start_time = time.time()
        return TimingContext(self, name, split=split, subset=subset)

    def add_phase(
        self,
        name: str,
        start: float,
        end: float,
        split: str | None = None,
        subset: str | None = None,
    ):
        """Adds a phase timing record."""
        if self._start_time is None:
            self._start_time = start

        phase: PhaseTiming = {
            "name": name,
            "start": start - self._start_time,
            "end": end - self._start_time,
        }
        if split:
            phase["split"] = split
        if subset:
            phase["subset"] = subset

        self.phases.append(phase)

    def quick_plot(self) -> None:
        """Plots a text-based bar chart of the recorded timing phases.

        When phases have ``subset`` or ``split`` metadata, the row label is
        prefixed with ``<split>/<subset>`` so multi-lingual / multi-split runs
        are easy to read at a glance.
        """
        if not self.phases:
            logger.info("No timing phases recorded.")
            return

        total_time = max(p["end"] for p in self.phases)
        if total_time == 0:
            return

        bar_length = 50

        def _label(phase: PhaseTiming) -> str:
            parts = []
            if "split" in phase:
                parts.append(phase["split"])
            if "subset" in phase:
                parts.append(phase["subset"])
            if parts:
                return "/".join(parts) + "/" + phase["name"]
            return phase["name"]

        labels = [_label(p) for p in self.phases]
        max_label_len = max(len(la) for la in labels)

        prev_group: tuple[str | None, str | None] = (None, None)
        for phase, label in zip(self.phases, labels):
            cur_group = (phase.get("split"), phase.get("subset"))
            if cur_group != prev_group and prev_group != (None, None):
                logger.info("")
            prev_group = cur_group

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
            logger.info(f"{label:<{max_label_len}} |{bar}| {duration:.1f}s")

        tracked_time = sum(p["end"] - p["start"] for p in self.phases)
        untracked = total_time - tracked_time
        logger.info(
            f"{' ' * max_label_len}  {total_time:.1f}s (untracked: {untracked:.1f}s)"
        )

