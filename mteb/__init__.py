from __future__ import annotations

from importlib.metadata import version

from mteb.benchmarks import (
    MTEB_MAIN_EN,
    MTEB_MAIN_RU,
    MTEB_RETRIEVAL_LAW,
    MTEB_RETRIEVAL_WITH_INSTRUCTIONS,
)
from mteb.evaluation import *
from mteb.overview import TASKS_REGISTRY, get_task, get_tasks

from .models import get_model, get_model_meta

__version__ = version("mteb")  # fetch version from install metadata


__all__ = [
    "MTEB_MAIN_EN",
    "MTEB_MAIN_RU",
    "MTEB_RETRIEVAL_LAW",
    "MTEB_RETRIEVAL_WITH_INSTRUCTIONS",
    "TASKS_REGISTRY",
    "get_tasks",
    "get_task",
    "get_model",
    "get_model_meta",
]
