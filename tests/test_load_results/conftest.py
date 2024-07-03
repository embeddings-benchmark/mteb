from __future__ import annotations

import os
from collections import defaultdict
from pathlib import Path
from typing import Any, List, Sequence, Type

import numpy as np
import torch

from mteb import AbsTask, Encoder


def all_subclasses(cls: Type[AbsTask]) -> List[Type[AbsTask]]:
    return sorted(
        list(
            set(cls.__subclasses__()).union(
                [s for c in cls.__subclasses__() for s in all_subclasses(c)]
            )
        ),
        key=lambda x: x.__name__,
    )


class MockEncoder(Encoder):
    def encode(
        self, sentences: Sequence[str], *, prompt_name: str | None = None, **kwargs: Any
    ) -> torch.Tensor | np.ndarray:
        return np.random.randn(len(sentences), 2)


def find_root_dir(starting_path: Path) -> Path:
    current_path = starting_path.resolve()
    while not (current_path / "pyproject.toml").exists():
        current_path = current_path.parent
    return current_path.absolute()


def get_all_tasks_results():
    root_dir = find_root_dir(Path(__file__))
    results_dir = root_dir / "results"
    task_files = defaultdict(list)
    for path, _, files in os.walk(results_dir):
        for file_name in files:
            if not file_name.endswith(".json"):
                continue
            task_name = file_name.split("/")[-1][:-5]
            task_files[task_name].append(os.path.join(path, file_name))
    return task_files
