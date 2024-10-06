import pytest

from mteb.overview import TASKS_REGISTRY
from .task_grid import MOCK_TASK_REGISTRY
import mteb.overview

@pytest.fixture(autouse=True)
def mock_mteb_get_task(monkeypatch):
    monkeypatch.setattr(mteb.overview, "TASKS_REGISTRY", MOCK_TASK_REGISTRY)