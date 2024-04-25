from __future__ import annotations

from importlib.metadata import version

from mteb.benchmarks import (
    MTEB_MAIN_EN,
    MTEB_RETRIEVAL_LAW,
    MTEB_RETRIEVAL_WITH_INSTRUCTIONS,
)
from mteb.evaluation import *
from mteb.get_tasks import get_tasks

__version__ = version("mteb")  # fetch version from install metadata


__all__ = [
    "MTEB_MAIN_EN",
    "MTEB_RETRIEVAL_LAW",
    "MTEB_RETRIEVAL_WITH_INSTRUCTIONS",
    "get_tasks",
]
