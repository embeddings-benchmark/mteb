"""Tests to ensure the superseded_by field is correct"""

from __future__ import annotations

import logging

import mteb
from mteb.get_tasks import _TASKS_REGISTRY

logging.basicConfig(level=logging.INFO)


def test_superseded_dataset_exists():
    tasks = mteb.get_tasks(exclude_superseded=False)
    for task in tasks:
        if task.superseded_by:
            assert task.superseded_by in _TASKS_REGISTRY, (
                f"{task} is superseded by {task.superseded_by} but {task.superseded_by} is not in the TASKS_REGISTRY"
            )
