"""Shared configuration and utilities for MVEB paper scripts."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

import mteb
from mteb.cache import ResultCache

RESULTS_PATH = Path("/Users/samoed/Desktop/results")
OUT_DIR = Path(__file__).parent

MVEB_BENCHMARK = "MVEB"

# Counterpart benchmarks for cross-modality comparison (name → scatter colour)
COUNTERPART_COLORS: dict[str, str] = {
    "MTEB(eng)": "#4e79a7",
    "MIEB(Img)": "#59a14f",
    "MAEB(beta)": "#f28e2b",
}

# MVEB task-type order, colours and display labels
TYPE_ORDER: list[str] = [
    "VideoClassification",
    "VideoZeroshotClassification",
    "VideoPairClassification",
    "VideoClustering",
    "VideoCentricQA",
    "Any2AnyRetrieval",
]

TYPE_COLORS: dict[str, str] = {
    "Any2AnyRetrieval": "#4e79a7",
    "VideoClassification": "#e15759",
    "VideoClustering": "#f28e2b",
    "VideoPairClassification": "#76b7b2",
    "VideoZeroshotClassification": "#59a14f",
    "VideoCentricQA": "#edc948",
}

TYPE_ABBREV: dict[str, str] = {
    "Any2AnyRetrieval": "Retrieval",
    "VideoClassification": "Classification",
    "VideoClustering": "Clustering",
    "VideoPairClassification": "Pair Classification",
    "VideoZeroshotClassification": "Zero-shot Classification",
    "VideoCentricQA": "Video-centric QA",
}

cache = ResultCache(cache_path=RESULTS_PATH)


def load_benchmark_means(benchmark_name: str) -> pd.Series:
    """Per-model mean score (0–1) across all available tasks in the benchmark."""
    bench = mteb.get_benchmark(benchmark_name)
    tasks = bench.tasks
    results = cache.load_results(tasks=tasks, require_model_meta=False)
    df = results.to_dataframe().set_index("task_name")
    available = [t.metadata.name for t in tasks if t.metadata.name in df.index]
    if not available:
        return pd.Series(dtype=float)
    return df.loc[available].mean(axis=0, skipna=True).dropna()


def fetch_model_meta(name: str) -> dict:
    try:
        m = mteb.get_model_meta(name)
        return {"n_parameters": m.n_parameters}
    except Exception:
        return {"n_parameters": None}


def short_name(model: str) -> str:
    return model.split("/")[-1]
