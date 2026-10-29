from __future__ import annotations

from typing import Any

import pytest
from datasets import Dataset

from mteb.mocks.mock_tasks.classification import MockClassificationTask


class _CountingColumn:
    """A label column that records how many times it is indexed row by row."""

    def __init__(self, values: list[Any]) -> None:
        self.values = list(values)
        self.reads = 0

    def __len__(self) -> int:
        return len(self.values)

    def __getitem__(self, index: int) -> Any:
        self.reads += 1
        return self.values[index]

    def __iter__(self) -> Any:
        return iter(self.values)


class _FakeDataset:
    """Minimal stand-in for `datasets.Dataset` used by `_undersample_data`."""

    def __init__(self, labels: list[Any]) -> None:
        self.labels = _CountingColumn(labels)
        self.selected: list[int] | None = None

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, key: str) -> _CountingColumn:
        assert key == "label"
        return self.labels

    def select(self, indices: list[int]) -> _FakeDataset:
        out = _FakeDataset([self.labels.values[i] for i in indices])
        out.selected = indices
        return out


def _task(**kwargs: Any) -> MockClassificationTask:
    task = MockClassificationTask()
    task.samples_per_label = kwargs.get("samples_per_label", 3)
    return task


def _reference_undersample(labels: list[Any], idxs: list[int], samples_per_label: int) -> list[int]:
    seen: dict[Any, int] = {}
    selected = []
    for i in idxs:
        label = labels[i]
        if seen.get(label, 0) < samples_per_label:
            selected.append(i)
            seen[label] = seen.get(label, 0) + 1
    return selected


@pytest.mark.parametrize("labels", [[i % 3 for i in range(30)], [str(i % 3) for i in range(30)]])
def test_undersample_data_matches_reference(labels: list[Any]) -> None:
    """The sampled indices must match a straightforward reference implementation."""
    task = _task()
    dataset = Dataset.from_dict({"text": [str(i) for i in range(len(labels))], "label": labels})

    _, idxs, selected = task._undersample_data(dataset, 0)

    assert selected == _reference_undersample(labels, idxs, task.samples_per_label)


def test_undersample_data_is_deterministic() -> None:
    task = _task(samples_per_label=2)
    dataset = Dataset.from_dict({"text": [str(i) for i in range(40)], "label": [i % 4 for i in range(40)]})

    first = task._undersample_data(dataset, 1)[2]
    second = task._undersample_data(dataset, 1)[2]

    assert first == second


@pytest.mark.parametrize("samples_per_label", [1, 3, 50])
def test_undersample_data_reads_label_column_once(samples_per_label: int) -> None:
    """Regression: the label column must not be indexed per row.

    `datasets.Column.__getitem__` builds a one-column table each call, which made this loop
    take minutes on 20k-row classification splits.
    """
    task = _task(samples_per_label=samples_per_label)
    dataset = _FakeDataset([i % 5 for i in range(200)])

    task._undersample_data(dataset, 0)  # type: ignore[arg-type]

    assert dataset.labels.reads == 0
