"""test that the mteb.MTEB works as intended and that encoders are correctly called and passed the correct arguments."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

import mteb

logging.basicConfig(level=logging.INFO)


@pytest.mark.parametrize("model", [mteb.get_model("baseline/random-encoder-baseline")])
def test_run_using_benchmark(model: mteb.EncoderProtocol, tmp_path: Path):
    """Test that a benchmark object can be run using the MTEB class."""
    bench = mteb.Benchmark(
        name="test_bench", tasks=mteb.get_tasks(tasks=["STS12", "SummEval"])
    )

    eval = mteb.MTEB(tasks=[bench])
    eval.run(
        model, output_folder=tmp_path.as_posix(), overwrite_results=True
    )  # we just want to test that it runs


@pytest.mark.parametrize("model", [mteb.get_model("baseline/random-encoder-baseline")])
def test_run_using_list_of_benchmark(model: mteb.EncoderProtocol, tmp_path: Path):
    """Test that a list of benchmark objects can be run using the MTEB class."""
    bench = [
        mteb.Benchmark(
            name="test_bench", tasks=mteb.get_tasks(tasks=["STS12", "SummEval"])
        )
    ]

    eval = mteb.MTEB(tasks=bench)
    eval.run(
        model, output_folder=tmp_path.as_posix(), overwrite_results=True
    )  # we just want to test that it runs
