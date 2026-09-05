from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from functools import cached_property

import pytest

import mteb
from mteb.abstasks import AbsTask

TaskKey = str | type[AbsTask]


def _resolve_task(key: TaskKey) -> AbsTask:
    """A registered task name (via ``mteb.get_task``) or a task class to instantiate."""
    if isinstance(key, type):
        return key()
    return mteb.get_task(key)


@dataclass
class ModelInfo:
    """A model baseline plus the expected final score for each task it is run on.

    Args:
        name: Identifier passed to ``loader``.
        expected_scores: Maps a task (registered name or task class) to the expected
            ``TaskResult.get_score()``. Iterating the ``ModelInfo`` yields one
            parametrize case per entry.
        loader: Builds the model from ``name``. Defaults to ``mteb.get_model``; pass a
            library constructor (e.g. ``SentenceTransformer``) to test a raw model.
    """

    name: str
    expected_scores: dict[TaskKey, float]
    loader: Callable[[str], mteb.EncoderProtocol] = field(
        default=mteb.get_model, repr=False
    )

    @cached_property
    def model(self) -> mteb.EncoderProtocol:
        """The loaded model, built once and shared by every case."""
        return self.loader(self.name)

    def __iter__(self) -> Iterator[pytest.ParameterSet]:
        for key, expected_score in self.expected_scores.items():
            task = _resolve_task(key)
            yield pytest.param(
                self.model,
                task,
                expected_score,
                id=f"{self.name}-{task.metadata.name}",
            )


def assert_final_score(
    model: mteb.EncoderProtocol, task: AbsTask, expected_score: float
) -> None:
    """Evaluate ``model`` on a fresh copy of ``task`` and assert its final score."""
    modalities = set(task.metadata.modalities)
    if {"image", "video"} & modalities:
        pytest.importorskip(
            "torchvision", reason="Vision dependencies are not installed"
        )
    if {"audio", "video"} & modalities:
        pytest.importorskip("torchaudio", reason="Audio dependencies are not installed")

    # Parametrized task objects are created at collection time and may be reused across
    # tests on the same xdist worker; evaluate a fresh task so scores never depend on
    # test order.
    task = type(task)()
    result = mteb.evaluate(model, task, cache=None)[0]
    assert result.get_score() == pytest.approx(expected_score, abs=1e-5, nan_ok=True), (
        f"{model} final score changed for {result.task_name}"
    )
