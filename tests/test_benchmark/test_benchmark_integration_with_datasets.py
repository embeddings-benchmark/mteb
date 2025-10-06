"""test mteb.MTEB's integration with datasets"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

import mteb
from mteb import MTEB
from mteb.abstasks import AbsTask
from mteb.cache import ResultCache
from mteb.results.generate_model_card import generate_model_card

from .mock_models import MockNumpyEncoder
from .task_grid import TASK_TEST_GRID

logging.basicConfig(level=logging.INFO)


@pytest.mark.parametrize("task", TASK_TEST_GRID)
@pytest.mark.parametrize("model", [MockNumpyEncoder()])
def test_benchmark_datasets(task: str | AbsTask, model: mteb.Encoder, tmp_path: Path):
    """Test that a task can be fetched and run"""
    eval = MTEB(tasks=[task])
    eval.run(model, output_folder=tmp_path.as_posix(), overwrite_results=True)

    # ensure that we can generate a readme from the output folder
    results_cache = ResultCache(tmp_path)
    output_path = tmp_path / "model_card.md"
    generate_model_card(
        model.mteb_model_meta.name,
        results_cache=results_cache,
        output_path=output_path,
    )
    assert output_path.exists()
