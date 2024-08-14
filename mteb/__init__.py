from __future__ import annotations

from importlib.metadata import version

from mteb.benchmarks import (
    MTEB_MAIN_EN,
    MTEB_MAIN_RU,
    MTEB_RETRIEVAL_LAW,
    MTEB_RETRIEVAL_WITH_INSTRUCTIONS,
    CoIR,
)

from mteb.abstasks import AbsTask, TaskMetadata
from mteb.encoder_interface import Encoder, EncoderWithConversationEncode, EncoderWithQueryCorpusEncode, EncoderWithSimilarity
from mteb.evaluation import *
from mteb.load_results import load_results, MTEBResults
from mteb.models import get_model, get_model_meta
from mteb.overview import TASKS_REGISTRY, get_task, get_tasks

from .benchmarks import Benchmark

__version__ = version("mteb")  # fetch version from install metadata


__all__ = [
    "AbsTask",
    "Encoder",
    "EncoderWithConversationEncode",
    "EncoderWithQueryCorpusEncode",
    "EncoderWithSimilarity",
    "MTEB_MAIN_EN",
    "MTEB_MAIN_RU",
    "MTEB_RETRIEVAL_LAW",
    "MTEB_RETRIEVAL_WITH_INSTRUCTIONS",
    "MTEBResults",
    "CoIR",
    "TASKS_REGISTRY",
    "TaskMetadata",
    "get_tasks",
    "get_task",
    "get_model",
    "get_model_meta",
    "load_results",
    "Benchmark",
]
