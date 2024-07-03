from __future__ import annotations

import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from mteb import Encoder


class MockEncoder(Encoder):
    def encode(
        self, sentences: Sequence[str], *, prompt_name: str | None = None, **kwargs: Any
    ) -> torch.Tensor | np.ndarray:
        return np.random.randn(len(sentences), 2)


def get_all_tasks_results():
    root_dir = Path(__file__).parent.parent.absolute()
    results_dir = root_dir / "results"
    task_files = defaultdict(list)
    for path, _, files in os.walk(results_dir):
        for file_name in files:
            if not file_name.endswith(".json"):
                continue
            task_name = file_name.split("/")[-1][:-5]
            task_files[task_name].append(os.path.join(path, file_name))
    return task_files
