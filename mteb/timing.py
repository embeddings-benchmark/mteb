from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

from typing_extensions import TypedDict

if TYPE_CHECKING:
    from types import TracebackType
logger = logging.getLogger(__name__)


class PhaseTiming(TypedDict):
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
        log_message: str | None = None,
    ):
        self.stack = stack
        self.name = name
        self.split = split
        self.subset = subset
        self.log_message = log_message
        self.start = 0.0

    def __enter__(self):
        if self.log_message:
            logger.info(self.log_message)
        self.start = time.monotonic()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ):
        end = time.monotonic()
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
        self,
        name: str,
        split: str | None = None,
        subset: str | None = None,
        log_message: str | None = None,
    ) -> TimingContext:
        """Returns a TimingContext for the specified phase name."""
        if self._start_time is None:
            self._start_time = time.monotonic()
        return TimingContext(
            self, name, split=split, subset=subset, log_message=log_message
        )

    def add_phase(
        self,
        name: str,
        start: float,
        end: float,
        split: str,
        subset: str,
    ):
        """Adds a phase timing record."""
        if self._start_time is None:
            self._start_time = start

        phase = PhaseTiming(
            name=name,
            start=start - self._start_time,
            end=end - self._start_time,
            split=split,
            subset=subset,
        )
        self.phases.append(phase)

    def quick_plot(self) -> str:
        """Plots a text-based bar chart of the recorded timing phases.

        When phases have ``subset`` or ``split`` metadata, the row label is
        prefixed with ``<split>/<subset>`` so multi-lingual / multi-split runs
        are easy to read at a glance.
        """
        if not self.phases:
            logger.info("No timing phases recorded.")
            return "No timing phases recorded."

        total_time = max(p["end"] for p in self.phases)
        if total_time == 0:
            return ""

        def _get_label(p: PhaseTiming) -> str:
            parts: list[str] = []
            if p.get("split"):
                parts.append(p["split"])
            if p.get("subset"):
                parts.append(p["subset"])
            return "/".join(parts + [p["name"]]) if parts else p["name"]

        labels = [_get_label(p) for p in self.phases]
        max_label_len = max(len(la) for la in labels)
        bar_length = 50
        lines = []
        prev_group: tuple[str | None, str | None] = (None, None)

        for phase, label in zip(self.phases, labels):
            cur_group = (phase.get("split"), phase.get("subset"))
            if prev_group not in {cur_group, (None, None)}:
                lines.append("")
            prev_group = cur_group

            duration = phase["end"] - phase["start"]
            s_pos = int((phase["start"] / total_time) * bar_length)
            e_pos = int((phase["end"] / total_time) * bar_length)
            b_len = max(1, e_pos - s_pos)
            bar = " " * s_pos + "█" * b_len + " " * (bar_length - s_pos - b_len)
            lines.append(f"{label:<{max_label_len}} |{bar}| {duration:.1f}s")

        tracked = sum(p["end"] - p["start"] for p in self.phases)
        lines.append(
            f"{' ' * max_label_len}  {total_time:.1f}s (untracked: {max(0.0, total_time - tracked):.1f}s)"
        )
        output = "\n".join(lines)
        logger.info("\n" + output)
        return output
