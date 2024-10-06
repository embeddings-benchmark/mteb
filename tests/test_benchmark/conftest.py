from __future__ import annotations

import pytest

import mteb

from .task_grid import MOCK_TASK_REGISTRY


@pytest.fixture
def mock_mteb_get_task(monkeypatch):
    monkeypatch.setattr(mteb.overview, "TASKS_REGISTRY", MOCK_TASK_REGISTRY)
